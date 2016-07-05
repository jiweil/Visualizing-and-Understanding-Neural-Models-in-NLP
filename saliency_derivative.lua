require "cutorch"
require 'cunn'
require "nngraph"

local model={};
paramx={}
paramdx={}
ada={}
local LookupTable=nn.LookupTable;

local params={batch_size=1,
    dimension=60,
    dropout=0.2,
    init_weight=0.1,
    max_length=100,
    vocab_size=19538
}

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

local function backward(Word,Delete,dh_left,dh_right)
    local dc_left=torch.zeros(Word:size(1),params.dimension):cuda();
    for t=Word:size(2),1,-1 do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
            c_t_1=model.c_left[t-1];
        end
        dh_left,dc_left=unpack(model.lstms_left[t]:backward({h_t_1,c_t_1,Word:select(2,t)},{dh_left,dc_left}));
        if Delete[t]:nDimension()~=0 then
            dh_left:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            dc_left:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
    local dc_right=torch.zeros(Word:size(1),params.dimension):cuda();
    for t=Word:size(2),1,-1 do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_right[t-1];
            c_t_1=model.c_right[t-1];
        end
        dh_right,dc_right=unpack(model.lstms_right[t]:backward({h_t_1,c_t_1,Word:select(2,t)},{dh_right,dc_right}));
        if Delete[t]:nDimension()~=0 then
            dh_right:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            dc_right:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
end


local function lstm(x,prev_h,prev_c)
    local drop_x=nn.Dropout(params.dropout)(x)
    local drop_h=nn.Dropout(params.dropout)(prev_h)
    local i2h=nn.Linear(params.dimension,4*params.dimension)(drop_x);
    local h2h=nn.Linear(params.dimension,4*params.dimension)(drop_h);
    local gates=nn.CAddTable()({i2h,h2h});
    local reshaped_gates =  nn.Reshape(4,params.dimension)(gates);
    local sliced_gates = nn.SplitTable(2)(reshaped_gates);
    local in_gate= nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform= nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate= nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate= nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
    local l1=nn.CMulTable()({forget_gate, prev_c})
    local l2=nn.CMulTable()({in_gate, in_transform})
    local next_c=nn.CAddTable()({l1,l2});
    local next_h= nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    return next_h,next_c
end

local function encoder_()
    local x_index=nn.Identity()();
    local prev_c=nn.Identity()();
    local prev_h=nn.Identity()();
    local x=LookupTable(params.vocab_size,params.dimension)(x_index);
    next_h,next_c=lstm(x,prev_h,prev_c)
    inputs={prev_h,prev_c,x_index};
    local module= nn.gModule(inputs,{next_h,next_c});
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module:cuda();
end


local function softmax_()
    ---softmax to get label
    local h_left=nn.Identity()();
    local h_right=nn.Identity()();
    local h2y_left= nn.Linear(params.dimension,5):noBias()(h_left)
    local h2y_right= nn.Linear(params.dimension,5):noBias()(h_right)
    local h=nn.CAddTable()({h2y_left,h2y_right});
    local pred= nn.LogSoftMax()(h)
    local module= nn.gModule({h_left,h_right,y},{pred});
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module:cuda()
end 


local function saliency_()
    --to get saliency value. 
    local h_left=nn.Identity()();
    local h_right=nn.Identity()();
    local h_left_vector=nn.Identity()();
    local h_right_vector=nn.Identity()();
    local left_value=nn.MM()({h_left_vector,h_left})
    local right_value=nn.MM()({h_right_vector,h_right});
    local h=nn.CAddTable()({left_value,right_value});
    local score=nn.Exp()(h)
    local module= nn.gModule({h_left,h_right,h_left_vector,h_right_vector},{score});
    return module:cuda()
end 

local function forward(Word,Word_r,Delete)
    Word=Word:cuda()
    for t=1,Word:size(2) do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
            c_t_1=model.c_left[t-1];
        end
        inputs={h_t_1,c_t_1,Word:select(2,t)}
        model.lstms_left[t]:evaluate()
        model.h_left[t],model.c_left[t]=unpack(model.lstms_left[t]:forward(inputs))
        if Delete[t]:nDimension()~=0 then
            model.h_left[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            model.c_left[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
    Word=Word_r:cuda()
    for t=1,Word:size(2) do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_right[t-1];
            c_t_1=model.c_right[t-1];
        end
        inputs={h_t_1,c_t_1,Word_r:select(2,t)}
        model.lstms_right[t]:evaluate()
        model.h_right[t],model.c_right[t]=unpack(model.lstms_right[t]:forward(inputs))
        if Delete[t]:nDimension()~=0 then
            model.h_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            model.c_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
end



cutorch.setDevice(1)
data=require("sentiment_bidi/data_")

encoder_left =encoder_()
encoder_right =encoder_()
saliency=saliency_()
softmax=softmax_();

encoder_left:getParameters()
encoder_right:getParameters()
softmax:getParameters()

paramx[1],paramdx[1]=encoder_left:parameters()
paramx[2],paramdx[2]=encoder_right:parameters()
paramx[3],paramdx[3] =softmax:parameters()

local file=torch.DiskFile("sentiment_bidi/model","r"):binary();
paramx_=file:readObject()
file:close();
for i=1,#paramx do 
    for j=1,#paramx[i] do
        paramx[i][j]:copy(paramx_[i][j]);
    end
end

model.h_left={};
model.c_left={};
model.h_right={};
model.c_right={};

model.lstms_left=g_cloneManyTimes(encoder_left,params.max_length)
model.lstms_right=g_cloneManyTimes(encoder_right,params.max_length)

local open_train_file=io.open("util/input_index.txt","r")
io.input(open_train_file)
End,Word,Word_r,Delete=data.read_train(open_train_file,params.batch_size);
-- get prediction first
forward(Word,Word_r,Delete);
pred=softmax:forward({model.h_left[Word:size(2)],model.h_right[Word:size(2)]})
local score,prediction=torch.max(pred,2)
local left_vector=paramx[3][1][{{prediction[1][1]},{}}]
local right_vector=paramx[3][2][{{prediction[1][1]},{}}]

value=saliency:forward({model.h_left[Word:size(2)]:t(),model.h_right[Word:size(2)]:t(),left_vector,right_vector})
local dh=saliency:backward({model.h_left[Word:size(2)]:t(),model.h_right[Word:size(2)]:t(),left_vector,right_vector},torch.Tensor({1}):cuda());
local dh_left=dh[1]
local dh_right=dh[2]
backward(Word,Delete,dh_left,dh_right)

paramdx[1][1]:add(paramdx[2][1]);
--paramdx[2][1] and paramdx[1][1] are word vector matrix
--we wish to get the derivative with respect to word vectors

local saliency=torch.Tensor(Word:size(2),params.dimension);
for i=1,Word:size(2) do 
    saliency[{{i},{}}]:copy(paramdx[1][1][{{tonumber(Word[1][i])},{}}])
end
saliency=torch.abs(saliency)

local file=io.open("matrix","w")
for i=1,saliency:size(1) do
    for j=1,saliency:size(2) do
        file:write(saliency[i][j].." ")
    end
    file:write("\n");
end
file:close()


require "cutorch"
require 'cunn'
require "nngraph"

local model={};
paramx={}
paramdx={}
ada={}
local LookupTable=nn.LookupTable;

local params={batch_size=1000,
    max_iter=20,
    dimension=60,
    dropout=0.2,
    train_file="../data/sequence_train.txt",
    --train_file="small",
    init_weight=0.1,
    learning_rate=0.05,
    dev_file="../data/sequence_dev_root.txt",
    test_file="../data/sequence_test_root.txt",
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
    local y=nn.Identity()();
    local h_left=nn.Identity()();
    local h_right=nn.Identity()();
    local h2y_left= nn.Linear(params.dimension,5):noBias()(h_left)
    local h2y_right= nn.Linear(params.dimension,5):noBias()(h_right)
    local h=nn.CAddTable()({h2y_left,h2y_right});
    local pred= nn.LogSoftMax()(h)
    local err= nn.ClassNLLCriterion()({pred, y})
    local module= nn.gModule({h_left,h_right,y},{err,pred});
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module:cuda()
end 

local function forward(Word,Word_r,Delete,isTraining)
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
        if isTraining then
            model.lstms_left[t]:training();
        else
            model.lstms_left[t]:evaluate();
        end
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
        if isTraining then
            model.lstms_right[t]:training();
        else
            model.lstms_right[t]:evaluate();
        end
        model.h_right[t],model.c_right[t]=unpack(model.lstms_right[t]:forward(inputs))
        if Delete[t]:nDimension()~=0 then
            model.h_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            model.c_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
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

local function test(filename)
    open_train_file=io.open(filename,"r")
    io.input(open_train_file)
    local End,Y,Word,Word_r,Delete;

    End=0;
    local right=0;
    local total=0;
    while End==0 do
        End,Y,Word,Word_r,Delete=data.read_train(open_train_file,params.batch_size);
        Y=Y:cuda()
        if End==1 then 
            break;
        end
        forward(Word,Word_r,Delete,false);
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()}))
        local score,prediction=torch.max(pred,2)
        prediction:resize(prediction:size(1)*prediction:size(2));
        total=total+prediction:size(1);
        for i=1,prediction:size(1) do
            if Y[i]==prediction[i] then
                right=right+1;
            end
        end
    end
    local accuracy=right/total;
    return accuracy;
end

cutorch.setDevice(1)
data=require("data")

encoder_left =encoder_()
encoder_right =encoder_()

softmax=softmax_();

encoder_left:getParameters()
encoder_right:getParameters()
softmax:getParameters()

paramx[1],paramdx[1]=encoder_left:parameters()
paramx[2],paramdx[2]=encoder_right:parameters()
paramx[3],paramdx[3] =softmax:parameters()

for i=1,3 do
    ada[i]={};
    for j=1,#paramx[i] do
        if paramx[i][j]:nDimension()==1 then
            ada[i][j]=1e-14*torch.rand(paramx[i][j]:size(1)):cuda()
        else
            ada[i][j]=1e-14*torch.rand(paramx[i][j]:size(1),paramx[i][j]:size(2)):cuda()
        end
    end
end

model.h_left={};
model.c_left={};
model.h_right={};
model.c_right={};

local timer=torch.Timer();
--embedding=data.read_embedding()

paramx[2][1]:copy(paramx[1][1])


model.lstms_left=g_cloneManyTimes(encoder_left,params.max_length)
model.lstms_right=g_cloneManyTimes(encoder_right,params.max_length)

iter=0;

store_param={};
for i=1,#paramx do
    store_param[i]={}
    for j=1,#paramx[i] do
        store_param[i][j]=torch.Tensor(paramx[i][j]:size());
        store_param[i][j]:copy(paramx[i][j]);
    end
end
local best_accuracy=-1;

while iter<params.max_iter do
    iter=iter+1;
    open_train_file=io.open(params.train_file,"r")
    io.input(open_train_file)
    local End,Y,Word,Delete;
    End=0;
    local time1=timer:time().real;
    while End==0 do
        for i=1,#paramdx do
            for j=1,#paramdx[i] do
                paramdx[i][j]:zero();
            end
        end
        End,Y,Word,Word_r,Delete=data.read_train(open_train_file,params.batch_size);
        if End==1 then 
            break;
        end
        forward(Word,Word_r,Delete,true)
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()}))
        local dh_left,dh_right=unpack(softmax:backward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()},{torch.ones(1):cuda(),torch.zeros(Word:size(1),5):cuda() }))
        backward(Word,Delete,dh_left,dh_right)
        for i=1,#ada do
            for j=1,#ada[i] do
                if i==2 and j==1 then
                    paramdx[i][j]:add(paramdx[1][j]);
                end
                ada[i][j]:add(torch.cmul(paramdx[i][j],paramdx[i][j]))
                paramx[i][j]:add(-torch.cdiv(paramdx[i][j],torch.sqrt(ada[i][j])):mul(params.learning_rate))
            end
        end
        paramx[1][1]:copy(paramx[2][1])
        --print(paramdx[1])
        --ada[2]=ada[2]:add(paramdx[2]:sqrt())
        --print(paramx[1])       
    end
    local time2=timer:time().real;
    acc_dev=test(params.dev_file)
    acc_test=test(params.test_file)
    if acc_test>best_accuracy then
        best_accuracy=acc_test;
        for i=1,#paramx do
            for j=1,#paramx[i] do
                store_param[i][j]:copy(paramx[i][j]);
            end
        end
    end
    if iter==20 then
        break;
    end
end

local file=torch.DiskFile("model","w"):binary();
file:writeObject(store_param);
file:close();

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
    learning_rate=0.1,
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

local function encoder_()
    local x_index=nn.Identity()();
    local prev_h=nn.Identity()();
    local x=LookupTable(params.vocab_size,params.dimension)(x_index);
    local drop_x=nn.Dropout(params.dropout)(x)
    local drop_h=nn.Dropout(params.dropout)(prev_h)
    local i2h=nn.Linear(params.dimension,params.dimension)(drop_x);
    local h2h=nn.Linear(params.dimension,params.dimension)(drop_h);
    local combine=nn.CAddTable()({i2h,h2h});;
    local next_h=nn.Tanh()(combine)
    inputs={prev_h,x_index};
    local module= nn.gModule(inputs,{next_h});
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module:cuda();
end


local function softmax_()
    local y=nn.Identity()();
    local h_left=nn.Identity()();
    local h2y_left= nn.Linear(params.dimension,5):noBias()(h_left)
    local pred= nn.LogSoftMax()(h2y_left)
    local err= nn.ClassNLLCriterion()({pred, y})
    local module= nn.gModule({h_left,y},{err,pred});
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return module:cuda()
end 

local function forward(Word,Delete,isTraining)
    Word=Word:cuda()
    for t=1,Word:size(2) do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
        end
        inputs={h_t_1,Word:select(2,t)}
        if isTraining then
            model.lstms_left[t]:training();
        else
            model.lstms_left[t]:evaluate();
        end
        model.h_left[t]=model.lstms_left[t]:forward(inputs)
        if Delete[t]:nDimension()~=0 then
            model.h_left[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
end

local function backward(Word,Delete,dh_left)
    for t=Word:size(2),1,-1 do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
        end
        dh_left=model.lstms_left[t]:backward({h_t_1,Word:select(2,t)},dh_left);
        dh_left=dh_left[1]
        if Delete[t]:nDimension()~=0 then
            dh_left:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
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
        forward(Word,Delete,false);
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],Y:cuda()}))
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

softmax=softmax_();

encoder_left:getParameters()
softmax:getParameters()

paramx[1],paramdx[1]=encoder_left:parameters()
paramx[2],paramdx[2] =softmax:parameters()

for i=1,2 do
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

local timer=torch.Timer();
--embedding=data.read_embedding()



model.lstms_left=g_cloneManyTimes(encoder_left,params.max_length)

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
        forward(Word,Delete,true)
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],Y:cuda()}))
        local deri=softmax:backward({model.h_left[Word:size(2)],Y:cuda()},{torch.ones(1):cuda(),torch.zeros(Word:size(1),5):cuda() })
        backward(Word,Delete,deri[1])
        for i=1,#ada do
            for j=1,#ada[i] do
                ada[i][j]:add(torch.cmul(paramdx[i][j],paramdx[i][j]))
                paramx[i][j]:add(-torch.cdiv(paramdx[i][j],torch.sqrt(ada[i][j])):mul(params.learning_rate))
            end
        end
        --print(paramdx[1])
        --ada[2]=ada[2]:add(paramdx[2]:sqrt())
        --print(paramx[1])       
    end
    local time2=timer:time().real;
    print(time2-time1)
    print("dev")
    acc_dev=test(params.dev_file)
    print("test")
    acc_test=test(params.test_file)
    print(acc_test)
    if acc_test>best_accuracy then
        best_accuracy=acc_test;
        for i=1,#paramx do
            for j=1,#paramx[i] do
                store_param[i][j]:copy(paramx[i][j]);
            end
        end
    end
    if iter==10 then
        break;
    end
end

local file=torch.DiskFile("model","w"):binary();
file:writeObject(store_param);
file:close();

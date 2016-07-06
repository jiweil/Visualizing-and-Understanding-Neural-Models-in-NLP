require "cutorch"
require "nn"
local stringx = require('pl.stringx')


local params={
    dimension=60,
}


cutorch.setDevice(1)

local file=torch.DiskFile("sentiment_uni/model","r"):binary();
paramx=file:readObject()
file:close();
local saliency;
local fr=io.open("util/input_index.txt","r")
io.input(fr)
while true do
    local str=io.read();
    print(str)
    if str==nil then
        End=1;
        break;
    end
    local list=stringx.split(stringx.strip(str)," ")
    print(list)
    local matrix=torch.Tensor(#list,params.dimension);
    for i=1,#list do 
        matrix[{{i},{}}]:copy(paramx[1][1][{{tonumber(list[i])},{}}])
    end
    local mean=torch.repeatTensor(torch.mean(matrix,1),#list,1);
    saliency=matrix-mean;
    saliency=nn.Square()(saliency)
end
fr:close()
local file=io.open("matrix","w")
for i=1,saliency:size(1) do
    for j=1,saliency:size(2) do
        file:write(saliency[i][j].." ")
    end
    file:write("\n");
end
file:close()

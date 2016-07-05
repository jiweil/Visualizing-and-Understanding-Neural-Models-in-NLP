import sys,re;
def ReadDic(dic_file):
    open_dic=open(dic_file,"r");
    dic={};
    index=0;
    for item in open_dic:
        item=item.strip();
        index=index+1;
        dic[item]=index
    return dic;
    
def Write(inputfile,outputfile,Dic):
    input=open(inputfile,"r");
    output=open(outputfile,"w");
    for line in input:
        G=re.split(" |\t",line.lower().strip());
        for item in G:  
            if item=="":continue;
            if Dic.has_key(item):
                output.write(str(Dic[item])+" ");
            else:
                output.write("1 ");
        output.write("\n")


dic=ReadDic(sys.argv[1]);
Write(sys.argv[2],sys.argv[3],dic);

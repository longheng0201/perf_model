from collections import deque
import math

class Node:
    def __init__(self, value,num):  #Thwc ; thw for 3d
        self.value = value
        self.idx = num
        self.optype=""
        self.next_nodes = []
        self.previous_nodes = []
        self.kernel=[]
        self.stride=[]
        self.dstride=[]
        self.pads=[]
        self.input_tensor=[]
        self.output_tensor=[]
        self.ibuf =0
        self.obuf =0
        self.wbuf =0
        self.ibuf_c =0  ##/16*4 in one R-core 
        self.obuf_c =0
        self.wbuf_c =0
        self.wbuf_loc=3 ##default ddr
        self.wbuf_loc_l=0 ##in ddr only 1 
        self.wbuf_valid=0
        self.core_wnum=1
        self.cluster_wnum=1
        

    def add_next_node(self, node):
        self.next_nodes.append(node) 
        node.previous_nodes.append(self)
    def cal_node_idepth (self,output_depth):
        max_input_shape =0
        big_than_oh = False
        max_output_shape=0
        for tensor in self.input_tensor:
            max_input_shape = max(max_input_shape,tensor.shape[0])
        for tensor in self.output_tensor:
            max_output_shape = max(max_output_shape,tensor.shape[0])
            if output_depth >= tensor.shape[0]:
                big_than_oh = True
    
        if(self.optype=="Reduce"):
            return max_input_shape
        else:
            if big_than_oh:
                return max_input_shape
            else:
                return (output_depth-1)*self.stride[0]+(self.dstride[0]-1)*(self.kernel[0]-1)+self.kernel[0]
    def mark_op_buffer(self):
        for tensor in self.input_tensor:
            tensor.buf =  tensor.depth * tensor.shape[1]*tensor.shape[2]/1024
            self.ibuf = self.ibuf + tensor.buf
            self.ibuf_c= self.ibuf_c + tensor.depth * tensor.shape[1]*math.ceil(tensor.shape[2]/16)*4 /1024
        ###############obuf
        for tensor in self.output_tensor:
            tensor.buf = tensor.depth * tensor.shape[1]*tensor.shape[2]/1024 
            self.obuf = self.obuf + tensor.buf
            self.obuf_c = self.obuf_c + tensor.depth * tensor.shape[1]*math.ceil(tensor.shape[2]/16)*4 /1024
        ############wgt
        if  self.optype=="Conv":
            self.wbuf = self.kernel[0]*self.kernel[1]*self.kernel[2]*self.kernel[3]/1024
            self.wbuf_c = self.kernel[0]*self.kernel[1]*math.ceil(self.kernel[2]/16)*4*self.kernel[3]/1024
        else:
            for tensor in self.output_tensor :
                self.wbuf = self.wbuf+ 4*tensor.shape[2]/1024
                 
            

class Tensor:
    def __init__(self, name):
        self.name = name
        self.buftype=""
        self.consumer= []
        self.producer= []
        self.shape =[]
        self.depth= 0  # lines
        self.loc= 3
        self.loc_l= 0
        self.loc_valid=0
        self.buf = 0 # self.depth*self.shape[1]*self.shape[2]/1024
        self.buf_c = 0 # self.depth*self.shape[1]*self.shape[2]/1024

class Graph:
    def __init__(self):
        self.nodes = []
        self.tensors = []
        self.network_ibuf_list ={} 
        self.network_obuf_list ={} 
        self.network_wbuf_list ={} 
        self.mem_cfg_left_list ={}
        self.valid_nodes = []
        self.f =""
    def add_node(self, node):
        self.nodes.append(node)
    def add_tensor(self, tensor):
        self.tensors.append(tensor)
    def find_node_by_value(self, value):
        for node in self.nodes:
            if node.value == value:
                return node
        return None
    def is_prefix_equal(self,list1, list2): #list1 in list2
        if len(list1) > len(list2):
            return False
        else:
            prefix_list2 = list2[:len(list1)]
            return (prefix_list2 == list1)
    def print_treewalker(self):
        for node in self.nodes:
            for next_node in node.next_nodes:
                print("node_name,",node.value,"next_node,",next_node.value)
        for node in self.valid_nodes:
            print("valid_node node_name,",node.value)
        for tensor in self.tensors:
                print("tensor_name,",tensor.name,"depth,",tensor.depth)
        print ("\nibuf",self.network_ibuf_list)
        print ("\nobuf",self.network_obuf_list )
        print ("\nwbuf",self.network_wbuf_list)
        print ("\nmem_cfg_left_list",self.mem_cfg_left_list)
    def dfs(self, current, end, valid_path,path, paths):
        if (current not in valid_path):
            pass
        else:
            path.append(current)  # 将当前节点添加到路径中

            if current == end:  # 如果当前节点是终点节点，则将路径添加到结果中
                paths.append(path[:])

            if current in self.nodes:  # 遍历当前节点的邻居节点
                for neighbor in current.next_nodes:
                    if neighbor not in path: ##avoid loop
                        self.dfs(neighbor, end,valid_path, path, paths)

            path.pop()  # 回溯，移除当前节点
    def find_all_paths(self, start, end,valid_nodes):
        paths = []  # 存储所有路径
        for idx in range(len(valid_nodes)):
            if valid_nodes[idx] == start:
                start_index = idx
            if valid_nodes[idx] == end:
                end_index = idx
        valid_path = valid_nodes[0:end_index+1]  ##from start will cause miss
        self.dfs(start, end,valid_path, [], paths)  # 调用深度优先搜索函数
        return paths
    def find_convenge(self, start, end):
        producer_paths = [] 
        for node in start.next_nodes:
            producer_path =[start,node]
            producer_paths.append(producer_path)
        paths = []  # 存储所有路径
        #self.dfs(start, end, valid_path,[], paths)  # 调用深度优先搜索函数
        paths = self.find_all_paths(start,end,self.valid_nodes )
        check_all_producer_path = True
        for producer_path in producer_paths:
            check_path =False
            for path in paths:
                is_prefix = self.is_prefix_equal(producer_path,path)
                if (is_prefix):
                    check_path=True
                    break
            if(check_path==False):
                check_all_producer_path = False
                break
        return check_all_producer_path  
    def cal_tensor_depth (self,path):  #A--->B  A output depth to B 
        inv_path = list(reversed(path)) 
        init_flag=True
        num = len(path)
        for node in inv_path:
            if(init_flag):
                prev_odepth = node.cal_node_idepth(1)
                init_flag=False
                num=num-1
            elif (num==1):
                prev_odepth = prev_odepth
                num=num-1
            else:
                cur_idepth = node.cal_node_idepth(prev_odepth)
                prev_odepth = cur_idepth
                num=num-1
        return prev_odepth
    
    
    def mark_buf_depth(self):   ## fro output view ;shourcut in output  view; all op gen 1 output 
        #####based on all op gen 1 output 
        for node in self.valid_nodes:
            print("mark node",node.value,"\n")
            index=self.valid_nodes.index(node)
            #node = self.find_node_by_value(self,node_name)
            if(len(node.next_nodes)==0):
                path_tmp =[node,node]
                for tensor in node.output_tensor:
                    tensor.buftype= "output"
                    tensor.depth= 1
            elif(node== self.valid_nodes[-1]): ##has been deal by output/input/ther nodes output
                pass 
            else:
                if (len(node.previous_nodes)==0):  ##special mark input
                    path_tmp =[node,node]
                    for tensor in node.input_tensor:
                        tensor.buftype= "input"
                        tensor.depth= self.cal_tensor_depth(path_tmp)
                #####################normal  output
                all_dot_depth=[]
                for key in self.valid_nodes[index+1:]:
                    paths = self.find_all_paths(node,key,self.valid_nodes)
                    print("-",key.value,"-")
                    dot_depth=[]
                    for path in paths:
                        depth = self.cal_tensor_depth(path)
                        dot_depth.append (depth)
                    all_dot_depth.extend(dot_depth)
                    convenge =False 
                    #if (key not in node.next_nodes): ##no need check next ,next convengeor not can be check rightly
                    convenge = self.find_convenge(node,key)
                    print("*")
                    if(convenge):
                        break
                max_depth = max(all_dot_depth)
                min_depth = min(all_dot_depth)
                for tensor in node.output_tensor:
                    if max_depth ==tensor.shape[0] :
                        if (tensor.buftype!=""):
                            raise ValueError("tensor re-marked error")
                        tensor.buftype= "layerwise"
                        tensor.depth= max_depth
                    elif min_depth ==max_depth:
                        if (tensor.buftype!=""):
                            raise ValueError("tensor re-marked error")
                        tensor.buftype= "direct"
                        tensor.depth= max_depth
                    else:
                        if (tensor.buftype!=""):
                            raise ValueError("tensor re-marked error")
                        tensor.buftype= "shorcut"
                        tensor.depth= max_depth
            print("\n key_name,",node.value)
        for tensor in self.tensors :
            if(tensor.buftype=="" and len(tensor.producer)==0 and len(tensor.consumer)==0):
               print(tensor.name)
               raise ValueError("tensor unmarked error") 
    def find_valid_nodes(self, initial_nodes):
        in_degrees = {}  # 存储节点的入度数
        valid_nodes = []  # 存储有效节点

        for node in self.nodes:
            in_degrees[node] = len(node.previous_nodes)
        # 初始化入度数为0
        # for node in initial_nodes:
        #     in_degrees[node.value] = 0

        # 计算所有节点的入度数
        #for node in initial_nodes:
        #    for next_node in node.next_nodes:
        #        in_degrees[next_node.value] = in_degrees.get(next_node.value, 0) + 1

        queue = deque()

        # 将入度为0的节点加入队列
        for node_value, degree in in_degrees.items():
            if degree == 0:
                queue.append(node_value)

        # 拓扑排序
        while queue:
            node_tmp = queue.popleft()
            valid_nodes.append(node_tmp)
            #node= self.find_node_by_value(node_value)
            # 更新与当前节点相邻节点的入度数
            for next_node in node_tmp.next_nodes:
                in_degrees[next_node] -= 1

                # 如果入度为0，则加入队列
                if in_degrees[next_node] == 0:
                    queue.appendleft(next_node)
        self.valid_nodes = valid_nodes
        return valid_nodes
    def mark_treewalker_buffer(self,mem_cfg):   
        network_ibuf=[]
        network_obuf=[]
        network_wbuf=[]
        network_pm_ibuf=[]
        network_lm_ibuf=[]
        network_gm_ibuf=[]
        network_pm_obuf=[]
        network_lm_obuf=[]
        network_gm_obuf=[]
        network_pm_wbuf=[]
        network_lm_wbuf=[]
        network_gm_wbuf=[]
        mem_cfg_left=[]
        pm_cfg_left=[]
        lm_cfg_left=[]
        gm_cfg_left=[]
        def get_pm_max_size(pm_cfg_left,risc_num,core_num,cluster_num):
            max_size=0
            index=0
            for i in range(core_num*cluster_num):
                if max_size<pm_cfg_left[i*risc_num] :
                    max_size=pm_cfg_left[i*risc_num]
                    index =i*risc_num
            return max_size,index
        def get_lm_max_size(lm_cfg_left,risc_num,core_num,cluster_num):
            max_size=0
            index=0
            for i in range(cluster_num * core_num):
                if max_size<lm_cfg_left[i] :
                    max_size=lm_cfg_left[i]
                    index   =i
            return max_size,index
        def get_gm_max_size(gm_cfg_left,risc_num,core_num,cluster_num):
            max_size=0
            index=0
            for i in range(cluster_num):
                if max_size<gm_cfg_left[i] :
                    max_size=gm_cfg_left[i]
                    index   =i
            return max_size,index
        [pm_size,lm_size,gm_size,ddr_size,risc_num,core_num,cluster_num] = mem_cfg
        ddr_idx =[0]
        ddr_size_list =[ddr_size]
        for i in range(risc_num*core_num*cluster_num):
            pm_cfg_left.append(pm_size)
            network_pm_ibuf.append(0)
            network_pm_obuf.append(0)
            network_pm_wbuf.append(0)
        for j in range(core_num*cluster_num):
            lm_cfg_left.append(lm_size)
            network_lm_ibuf.append(0)
            network_lm_obuf.append(0)
            network_lm_wbuf.append(0)
        for k in range(cluster_num):
            gm_cfg_left.append(gm_size)
            network_gm_ibuf.append(0)
            network_gm_obuf.append(0)
            network_gm_wbuf.append(0)
        mem_cfg_left.append(pm_cfg_left )
        mem_cfg_left.append(lm_cfg_left )
        mem_cfg_left.append(gm_cfg_left )   
        mem_cfg_left.append(ddr_size_list)
        network_ibuf.append(network_pm_ibuf)
        network_ibuf.append(network_lm_ibuf)
        network_ibuf.append(network_gm_ibuf)
        network_ibuf.append(ddr_idx)
        network_obuf.append(network_pm_obuf)
        network_obuf.append(network_lm_obuf)
        network_obuf.append(network_gm_obuf)
        network_obuf.append(ddr_idx)
        network_wbuf.append(network_pm_wbuf)
        network_wbuf.append(network_lm_wbuf)
        network_wbuf.append(network_gm_wbuf)
        network_wbuf.append(ddr_idx)
        marked_nodes = []
        index =0 
        debug_index =97
        for node in self.valid_nodes:
            ######cal wgt
            print("treewalker node",node.value,"__",index)
            node.mark_op_buffer()    ##put output 
            #####check  convenge  ###note   tensor muliinput has same size operation
            convenge_reduce =0
            if index ==debug_index:
                print(node.value)
            if (len(node.previous_nodes)!=0):
                for buf in node.input_tensor:
                    if buf.depth==buf.shape[0] :
                        convenge_reduce = 1
            print("convenge_reduce to this node",convenge_reduce,node.value)
            if convenge_reduce==1:
                for marked_node in marked_nodes:
                    check_convenge = False
                    if (marked_node not in node.previous_nodes):
                        check_convenge = self.find_convenge(marked_node,node)
                        if check_convenge:
                            network_wbuf[marked_node.wbuf_loc][marked_node.wbuf_loc_l] = network_wbuf[marked_node.wbuf_loc][marked_node.wbuf_loc_l] - marked_node.wbuf_valid * marked_node.wbuf
                            mem_cfg_left[marked_node.wbuf_loc][marked_node.wbuf_loc_l] = mem_cfg_left[marked_node.wbuf_loc][marked_node.wbuf_loc_l] + marked_node.wbuf_valid *marked_node.wbuf
                            marked_node.wbuf_valid = 0
                            for buf in marked_node.output_tensor:
                                network_obuf[buf.loc][buf.loc_l] = network_obuf[buf.loc][buf.loc_l] - buf.buf *buf.loc_valid
                                mem_cfg_left[buf.loc][buf.loc_l] = mem_cfg_left[buf.loc][buf.loc_l] + buf.buf *buf.loc_valid
                                buf.buf_valid = 0
                            if(len(marked_node.previous_nodes)==0):
                                for buf in marked_node.input_tensor:
                                    network_ibuf[buf.loc][buf.loc_l] = network_ibuf[buf.loc][buf.loc_l] - buf.buf* buf.loc_valid
                                    mem_cfg_left[buf.loc][buf.loc_l] = mem_cfg_left[buf.loc][buf.loc_l] + buf.buf* buf.loc_valid
                                    buf.buf_valid = 0
            ###################################input  only split ic ,no oc split tbd
            pm_cfg_left_max,pm_index = get_pm_max_size(mem_cfg_left[0],risc_num,core_num,cluster_num)
            lm_cfg_left_max,lm_index = get_lm_max_size(mem_cfg_left[1],risc_num,core_num,cluster_num)
            gm_cfg_left_max,gm_index = get_gm_max_size(mem_cfg_left[2],risc_num,core_num,cluster_num)
            if len(node.previous_nodes)==0:
                if node.ibuf_c <= pm_cfg_left_max :
                    for i in range(risc_num):
                        network_ibuf[0][pm_index+i] = network_ibuf[0][pm_index+i] + node.ibuf_c
                        mem_cfg_left[0][pm_index+i] = mem_cfg_left[0][pm_index+i] - node.ibuf_c
                    for tensor in node.input_tensor:
                        tensor.loc = 0
                        tensor.loc_l = pm_index
                        tensor.loc_valid = 1
                    node.core_wnum =1
                    node.cluster_num=1
                    
                elif(node.ibuf <= lm_cfg_left_max ):
                    network_ibuf[1][lm_index] = network_ibuf[1][lm_index] + node.ibuf
                    mem_cfg_left[1][lm_index] = mem_cfg_left[1][lm_index] - node.ibuf
                    for tensor in node.input_tensor:
                        tensor.loc = 1
                        tensor.loc_valid = 1
                        tensor.loc_l =lm_index
                    node.core_wnum=1
                    node.cluster_wnum=1
                elif(node.ibuf <= gm_cfg_left_max):
                    network_ibuf[2][gm_index] = network_ibuf[2][gm_index] + node.ibuf
                    mem_cfg_left[2][gm_index] = mem_cfg_left[2][gm_index] - node.ibuf
                    for tensor in node.input_tensor:
                        tensor.loc = 2
                        tensor.loc_valid = 1
                        tensor.loc_l =gm_index
                    node.core_wnum= core_num
                    node.cluster_wnum=1
                else:
                    network_ibuf[3][0] = network_ibuf[3][0] + node.ibuf
                    mem_cfg_left[3][0] = mem_cfg_left[3][0] - node.ibuf
                    for tensor in node.input_tensor:
                        tensor.loc = 3
                        tensor.loc_valid = 1
                        tensor.loc_l =0
                    node.core_wnum= core_num
                    node.cluster_wnum=cluster_num
            pm_cfg_left_max,pm_index = get_pm_max_size(mem_cfg_left[0],risc_num,core_num,cluster_num)
            lm_cfg_left_max,lm_index = get_lm_max_size(mem_cfg_left[1],risc_num,core_num,cluster_num)
            gm_cfg_left_max,gm_index = get_gm_max_size(mem_cfg_left[2],risc_num,core_num,cluster_num)
            ########first try output in inner one ;then output ;the wgt ;
            if node.obuf_c <= pm_cfg_left_max :
                for i in range(risc_num):
                    network_obuf[0][pm_index+i] = network_obuf[0][pm_index+i] + node.obuf_c
                    mem_cfg_left[0][pm_index+i] = mem_cfg_left[0][pm_index+i] - node.obuf_c
                for tensor in node.output_tensor:
                    tensor.loc = 0
                    tensor.loc_l = pm_index
                    tensor.loc_valid = 1
            elif(node.obuf <= lm_cfg_left_max ):
                network_obuf[1][lm_index] = network_obuf[1][lm_index] + node.obuf
                mem_cfg_left[1][lm_index] = mem_cfg_left[1][lm_index] - node.obuf
                for tensor in node.output_tensor:
                    tensor.loc = 1
                    tensor.loc_valid = 1
                    tensor.loc_l =lm_index
            elif(node.obuf <= gm_cfg_left_max):
                network_obuf[2][gm_index] = network_obuf[2][gm_index] + node.obuf
                mem_cfg_left[2][gm_index] = mem_cfg_left[2][gm_index] - node.obuf
                for tensor in node.output_tensor:
                    tensor.loc = 2
                    tensor.loc_valid = 1
                    tensor.loc_l =gm_index
            else:
                network_obuf[3][0] = network_obuf[3][0] + node.obuf
                mem_cfg_left[3][0] = mem_cfg_left[3][0] - node.obuf
                for tensor in node.output_tensor:
                    tensor.loc = 3
                    tensor.loc_valid = 1
                    tensor.loc_l =0
                    
            pm_cfg_left_max,pm_index = get_pm_max_size(mem_cfg_left[0],risc_num,core_num,cluster_num)
            lm_cfg_left_max,lm_index = get_lm_max_size(mem_cfg_left[1],risc_num,core_num,cluster_num)
            gm_cfg_left_max,gm_index = get_gm_max_size(mem_cfg_left[2],risc_num,core_num,cluster_num)
            ##################weight 
            if node.wbuf <= pm_cfg_left_max:
                for i in range(risc_num):
                    network_wbuf[0][pm_index+i] = network_wbuf[0][pm_index+i] + node.wbuf_c
                    mem_cfg_left[0][pm_index+i] = mem_cfg_left[0][pm_index+i] - node.wbuf_c
                node.wbuf_loc = 0
                node.wbuf_valid = 1
                node.wbuf_loc_l=pm_index
                node.core_wnum=1
                node.cluster_wnum=1
            elif(node.wbuf <= lm_cfg_left_max):
                network_wbuf[1][lm_index] = network_wbuf[1][lm_index] + node.wbuf
                mem_cfg_left[1][lm_index] = mem_cfg_left[1][lm_index] - node.wbuf
                node.wbuf_loc = 1
                node.wbuf_valid = 1
                node.wbuf_loc_l=lm_index
                node.core_wnum=1
                node.cluster_wnum=1
            elif(node.wbuf <= gm_cfg_left_max):
                network_wbuf[2][gm_index] = network_wbuf[2][gm_index] + node.wbuf
                mem_cfg_left[2][gm_index] = mem_cfg_left[2][gm_index] - node.wbuf
                node.wbuf_loc = 2
                node.wbuf_valid = 1
                node.wbuf_loc_l=gm_index
                node.core_wnum=core_num
                node.cluster_wnum=1
            else:
                network_wbuf[3][0] = network_wbuf[3][0] + node.wbuf
                mem_cfg_left[3][0] = mem_cfg_left[3][0] - node.wbuf
                node.wbuf_loc = 3
                node.wbuf_valid = 1
                node.wbuf_loc_l=0
                node.core_wnum=core_num
                node.cluster_wnum=cluster_num
            
            
            marked_nodes.append(node)
            self.network_ibuf_list[node.value] = network_ibuf.copy() 
            self.network_obuf_list[node.value] = network_obuf.copy() 
            self.network_wbuf_list[node.value] = network_wbuf.copy() 
            self.mem_cfg_left_list[node.value] = mem_cfg_left.copy() 
            index=index+1
            
            
    
                    
            
            
#############################################test for class################################
#创建图对象
def sanity_test():
    graph = Graph()

    # 创建节点对象
    A = Node('A',0)
    B = Node('B',0)
    C = Node('C',0)
    D = Node('D',0)
    E = Node('E',0)
    F = Node('F',0)
    G = Node('G',0)
    H = Node('H',0)
    I = Node('I',0)

    a0 = Tensor('a0')
    a1 = Tensor('a1')
    b0 = Tensor('b0')
    b1 = Tensor('b1')
    b2 = Tensor('b2')
    b3 = Tensor('b3')
    b4 = Tensor('b4')
    b5 = Tensor('b5')
    b6 = Tensor('b6')
    b7 = Tensor('b7')
    b8 = Tensor('b8')


    # 将节点添加到图中
    graph.add_node(A)
    graph.add_node(B)
    graph.add_node(C)
    graph.add_node(D)
    graph.add_node(E)
    graph.add_node(F)
    graph.add_node(G)
    graph.add_node(H)
    graph.add_node(I)


    graph.add_tensor(a0)
    graph.add_tensor(a1)
    graph.add_tensor(b0)
    graph.add_tensor(b1)
    graph.add_tensor(b2)
    graph.add_tensor(b3)
    graph.add_tensor(b4)
    graph.add_tensor(b5)
    graph.add_tensor(b6)
    graph.add_tensor(b7)
    graph.add_tensor(b8)

    # 建立节点之间的连接关系
    A.add_next_node(B)
    A.add_next_node(D)
    B.add_next_node(C)
    D.add_next_node(C)
    E.add_next_node(F)
    C.add_next_node(G)
    F.add_next_node(G)
    G.add_next_node(H)
    H.add_next_node(I)
    G.add_next_node(I)
    F.add_next_node(I)

    A.input_tensor.append(a0) 
    E.input_tensor.append(a1) 

    B.input_tensor.append(b0) 
    C.input_tensor.append(b1) 
    C.input_tensor.append(b3) 
    D.input_tensor.append(b0) 
    F.input_tensor.append(b4) 
    G.input_tensor.append(b2) 
    G.input_tensor.append(b5)
    H.input_tensor.append(b6)
    I.input_tensor.append(b7)
    I.input_tensor.append(b6)
    I.input_tensor.append(b5)

    A.output_tensor.append(b0) 
    B.output_tensor.append(b1) 
    C.output_tensor.append(b2) 
    D.output_tensor.append(b3) 
    E.output_tensor.append(b4) 
    F.output_tensor.append(b5) 
    G.output_tensor.append(b6) 
    H.output_tensor.append(b7) 
    I.output_tensor.append(b8) 


    A.optype="Conv"
    B.optype="Conv"
    C.optype="Add"
    D.optype="Conv"
    E.optype="Conv"
    F.optype="Conv"
    G.optype="Conv"
    H.optype="Conv"
    I.optype="ADD"
    ##################################[0,1,2]
    ###
    A.kernel= [3,1,3,3] 
    B.kernel= [3,1,3,3]
    C.kernel= [1,1,1,3]  
    D.kernel= [3,1,3,3]
    E.kernel= [3,1,3,3]
    F.kernel= [3,1,3,3]
    G.kernel= [3,1,3,3]
    H.kernel= [3,1,3,3]
    I.kernel= [1,1,1,3]

    A.stride= [1,1,1] 
    B.stride= [1,1,1]
    C.stride= [1,1,1]  
    D.stride= [1,1,1]
    E.stride= [1,1,1]
    F.stride= [1,1,1]
    G.stride= [1,1,1]
    H.stride= [1,1,1]
    I.stride= [1,1,1]

    A.dstride= [1,1,1] 
    B.dstride= [1,1,1]
    C.dstride= [1,1,1]  
    D.dstride= [1,1,1]
    E.dstride= [1,1,1]
    F.dstride= [1,1,1]
    G.dstride= [1,1,1]
    H.dstride= [1,1,1]
    I.dstride= [1,1,1]

    a0.shape= [10,10,10] 
    a1.shape= [10,10,10]
    b0.shape= [10,10,10] 
    b1.shape= [10,10,10]
    b2.shape= [10,10,10]  
    b3.shape= [10,10,10]
    b4.shape= [10,10,10]
    b5.shape= [10,10,10]
    b6.shape= [10,10,10]
    b7.shape= [10,10,10]
    b8.shape= [10,10,10]


    initial_nodes = [A,E]
    mem_cfg = [10,100,1000,1000000,4,4,4]   ## inner to outer

    valid_nodes = graph.find_valid_nodes(initial_nodes)
    graph.mark_buf_depth()
    graph.mark_treewalker_buffer(mem_cfg)
    graph.print_treewalker()


if __name__ == '__main__':
    sanity_test()
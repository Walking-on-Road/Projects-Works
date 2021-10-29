#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np #导入数组计算工具
import matplotlib.tri as tri #以下皆为绘图工具
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

print("王老师，这个程序的输出有点慢(professor wang,Please wait for some time)")
# In[65]:


class Node(object): #创建节点类
    def __init__(self,nnd,x,y):
        self.ID = nnd #第nnd个节点
        self.x = x
        self.y = y
        self.nodeAK = ("Ux","Uy")
        ## 位移情况
        self.ux = None
        self.uy = None  #定义属性为None是为了便于后续查找
        self.disp = dict.fromkeys(self.nodeAK,0.) #建立一个位移的字典。
        ## 受力情况
        self.Fx,self.Fy = None,None


# In[66]:


class Triangle(object): #三角形应力单元
    
    def __init__(self,nel,nodes): #初始类别属性
        self.E = 200e6   #弹性模量
        self.nu = 0.3 #泊松比
        self.t = 0.025 #厚度
        self.nodes = nodes
        self.ID = nel #单元的编号
        self.volume = None
        self.elementIk = (("sx","sy","sxy"))
        self.k = self.calcTri_Ke()
        self.stress =  dict.fromkeys(self.elementIk,0.)
        self.strain =  dict.fromkeys(self.elementIk,0.)
        self.u = None        
        
    def calcTri_D(self): #计算本构矩阵D
        E = self.E
        nu = self.nu
        a = E/(1-nu**2)
        self.D = a*np.array([[1.,nu,0.],
                        [nu,1.,0.],
                        [0.,0.,(1-nu)/2.]])
        return self.D
        
    def calcTri_B(self):  #计算应变矩阵B
        nodes = self.nodes
        x1,y1 = nodes[0].x,nodes[0].y
        x2,y2 = nodes[1].x,nodes[1].y
        x3,y3 = nodes[2].x,nodes[2].y
        area = ((x2*y3-x3*y2)+(y2-y3)*x1+(x3-x2)*y1)
        self.volume =1./2.*area #三角形面积
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1
        self.B = 1./area*np.array([[b1,0,b2,0,b3,0],
                                     [0.,c1,0,c2,0,c3],
                                     [c1,b1,c2,b2,c3,b3]])    
        return self.B
        
    def calcTri_Ke(self): #计算单元刚度矩阵
        self.calcTri_B()
        self.calcTri_D()
        self._Ke = self.t*self.volume*np.dot(np.dot(self.B.T,self.D),self.B)
        return self._Ke
  
    def assembel_into_K(self,K):
        ndIDs = [nd.ID for nd in element.nodes]
        self.calcTri_Ke()  #计算一个单元的刚度矩阵
        for N1,I in enumerate(ndIDs):
            for N2,J in enumerate(ndIDs):
                K[2*I:2*(I+1),2*J:2*(J+1)] += self.k[2*N1:2*(N1+1),2*N2:2*(N2+1)]
                
#通过节点位移计算三角形单元应力与应变
    def evaluate_stress_strain(self):
        u = np.array([[nd.disp[key] for nd in self.nodes for key in nd.nodeAK[:2]]])
        self._notdeal_stress = np.dot(np.dot(self.D,self.B),u.T)
        self._notdeal_strain = np.dot(self.B,u.T)
        n = len(self.elementIk)
        for i,val in enumerate(self.elementIk):
            self.stress[val] += self._notdeal_stress[i::n]
            self.strain[val] += self._notdeal_strain[i::n]
 


# In[67]:


################################################### main-function(主函数) ###############################################################
E = 200e6
nu = 0.3
t = 0.025
nc= np.loadtxt('a.txt') #nc:node coordinate,代表着节点的坐标
eln = np.loadtxt('b.txt'  ) #eln:element number,代表着三角形单元的编号
x,y = nc[:,0],nc[:,1] #得到节点的横坐标与纵坐标
nodes = [  ] 
for v,nd in enumerate(nc):
    #函数enumerate()可以同时获得节点坐标的索引号和对应的坐标值，也就是说nd获得第v个节点的横纵坐标
    n = Node(v,x[v],y[v]) #其中v代表着节点的索引值。
    nodes.append(n)  #创建节点
    
N = len(nodes)
nof = 2 # 节点自由度
K = np.zeros((N*2,N*2))
F = np.zeros(N*2)
    
#施加边界条件（确保左边固定约束，右边均匀载荷）
for nd in nodes:
    if nd.x == 0.:
        nd.ux = 0.
        nd.uy = 0.
    if nd.x == 0.1:
        F[2*nd.ID]= 0.5 # Fx作用在nd.ID上  

Els = [] #获得三角形单元
for v,el in enumerate(eln):
    a = int(el[0]-1)
    b = int(el[1]-1)
    c = int(el[2]-1)
    node1 = nodes[a]
    node2 = nodes[b]
    node3 = nodes[c]
    e = Triangle(v,(node1,node2,node3))
    Els.append(e)  #创建三角形单元
            
for element in Els:
    element.assembel_into_K(K) #调用Triangle类中的函数获得总体刚度矩阵


# In[68]:


# 生成U向量
# 注意位移的排列顺序
u = [[nd.ux,nd.uy] for nd in nodes]
U = np.array(u).flatten() #a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 

keeped_ind = np.where(U == None)[0] # 划行划线法
keeped = [row for row,val in enumerate(U) if val is None]
keeped_K = K[keeped_ind,:][:,keeped_ind]
keeped_F = F[keeped_ind]


# In[69]:


keeped_U = np.linalg.solve(keeped_K,keeped_F)#求解线性方程组
U[keeped_ind] = keeped_U # 所有节点的位移


# In[70]:


######################################################## 后处理(计算位移以及应力应变）) ##########################################################################
U_x = U[0::2].tolist() #节点的x轴方向位移量
U_y = U[1::2].tolist()  #节点的y轴方向位移量


# In[71]:


nodeAK = ("Ux","Uy")
for i,j in enumerate(keeped):
    A = j%2
    B = int(j/2)
    nodes[B].disp[nodeAK[A]] = keeped_U[i] #获取各个节点的相应位移
for el in Els:
    el.evaluate_stress_strain() #调用Triang类中的函数，计算出三角形单元的应力、应变。


# In[72]:


stress_x = np.array([el.stress["sx"][0][0] for el in Els])    #x轴方向上单元应力
stress_y = np.array([el.stress["sy"][0][0] for el in Els])    #y轴方向上单元应力
stress_xy = np.array([el.stress["sxy"][0][0] for el in Els])  #xy方向上的剪切应力
strain_x = np.array([el.strain["sx"][0][0] for el in Els])    #x轴方向上的单元应变
strain_y = np.array([el.strain["sy"][0][0] for el in Els])    #y轴方向上的单元应变
strain_xy = np.array([el.strain["sxy"][0][0] for el in Els])  #xy方向上的剪切应变


# In[73]:


############################################## 后处理-绘制云图（Post-processing - Cloud mapping）################################################################
'''绘制云图,Cloud mapping'''    
nx = [nd.x for nd in nodes]  #get_nodes()获得系统的节点列表
ny = [nd.y for nd in nodes]
nID = []
for element in Els:
    inds = [nd.ID for nd in element.nodes]
    nID.append(inds)
#get_elements获得系统的单元列表
tr = tri.Triangulation(nx,ny, nID)

fig1, fig2,fig3,fig4,fig5 = plt.figure(), plt.figure(),plt.figure(),plt.figure(),plt.figure()
fig6, fig7,fig8,fig9,fig10 = plt.figure(), plt.figure(),plt.figure(),plt.figure(),plt.figure()
ax1, ax2,ax3,ax4,ax5 = fig1.add_subplot(111), fig2.add_subplot(111),fig3.add_subplot(111),fig4.add_subplot(111),fig5.add_subplot(111)
ax6, ax7,ax8,ax9,ax10 = fig6.add_subplot(111), fig7.add_subplot(111),fig8.add_subplot(111),fig9.add_subplot(111),fig10.add_subplot(111)

'''节点图,node drawing'''
ax1.set_title("A graph of nodes in a triangular grid")
ax1.set_aspect("equal")
ax1.scatter(nc[:,0],nc[:,1]) #以self._pt的第一列所有元素为横坐标，self._pt的第二列所有元素为纵坐标。  

'''三角形网格划分图,Triangular meshes'''
patches = []       
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches, color="w", edgecolor="b", alpha=0.5)
ax2.set_title("Mesh")
ax2.add_collection(pc)
ax2.set_aspect("equal")
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 0.1])             
   
'''stress_xy切应力图'''
patches = []    #切应力图
#ex,ey = [],[]
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1)
pc.set_array(stress_xy)
ax3.add_collection(pc)
ax3.set_title("Stress_xy")
ax3.set_xlim([0, 0.1])
ax3.set_ylim([0, 0.1])
ax3.set_aspect("equal")
fig3.colorbar(pc) 
 
'''stress_x,x轴方向正应力图'''
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1) 
pc.set_array(stress_x)  #x正应力
ax4.add_collection(pc)
ax4.set_title("Stress_x")
ax4.set_xlim([0, 0.1])
ax4.set_ylim([0, 0.1])
ax4.set_aspect("equal")
fig4.colorbar(pc)      

'''stress_y,y轴方向正应力图'''
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1) 
pc.set_array(stress_y)  #y正应力
ax5.add_collection(pc)
ax5.set_title("Stress_y")
ax5.set_xlim([0, 0.1])
ax5.set_ylim([0, 0.1])
ax5.set_aspect("equal")
fig5.colorbar(pc) 

'''strain_xy,切应便变图'''
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1) 
pc.set_array(strain_xy)  #切应变
ax6.add_collection(pc)
ax6.set_title("Strain_xy")
ax6.set_xlim([0, 0.1])
ax6.set_ylim([0, 0.1])
ax6.set_aspect("equal")
fig6.colorbar(pc) 

'''strain_x,x轴正应变图'''
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1) 
pc.set_array(strain_x)  #x正应变
ax7.add_collection(pc)
ax7.set_xlim([0, 0.1])
ax7.set_title("Strain_x")
ax7.set_ylim([0, 0.1])
ax7.set_aspect("equal")
fig7.colorbar(pc) 

'''strain_y,y切应力图'''
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1) 
pc.set_array(strain_y)  #y正应变
ax8.add_collection(pc)
ax8.set_title("Strain_y")
ax8.set_xlim([0, 0.1])
ax8.set_ylim([0, 0.1])
ax8.set_aspect("equal")
fig8.colorbar(pc) 

'''水平位移云图,Horizontal displacement cloud map'''
ax9.set_title("Disp_X")
disp_X = ax9.tripcolor(tr,U_x,color="k",  cmap="jet",shading = 'gouraud') #水平位移云图
fig9.colorbar(disp_X)

'''竖直位移云图,Vertical displacement cloud map'''
ax10.set_title("Disp_Y")
disp_Y = ax10.tripcolor(tr, U_y, color="k", cmap="jet",shading = 'gouraud')  #竖直位移云图
fig10.colorbar(disp_Y)
   
plt.show()


# In[74]:
######################### ############################### Output the disp and stress,strain  ###########################################

print(U_x)
print(U_y)

# In[75]:


print(stress_x)
print(stress_y)
print(stress_xy)
print(strain_x)
print(strain_y)
print(strain_xy)

# In[76]:


max(U_x)


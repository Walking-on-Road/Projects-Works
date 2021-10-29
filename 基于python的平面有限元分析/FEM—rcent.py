#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.integrate import dblquad

print("此为有限元平面四边形程序，王老师，程序运行需要一会儿时间\n")
print("本程序共处理2500个四边形单元，模型为10*10的正方形，左边约束，右边均匀载荷\n")
print("学硕，彭扬，S200200268，（前面已经交了一次基于python的三角形单元）\n")


# In[50]:


class mesh(object):
    def __init__(self,xlim,ylim,nx,ny):
    #形参为x的坐标范围，y的坐标范围，x轴上点的个数，y轴上点的个数
        """
        把求解域用网格划分
        返回网格的节点，坐标，单元划分等
        """
        self._X = np.linspace(ylim[0],ylim[1],ny+1,endpoint=True, retstep=False, dtype=None) #在xlim[0],xlim[1]之间取nx+1个点，得到相应的数组。
        self._Y = np.linspace(ylim[0],ylim[1],ny+1,endpoint=True, retstep=False, dtype=None) #在ylim[0],ylim[1]之间取ny+1个点
        self._Nx = nx  #x轴上点的个数
        self._Ny = ny  #y轴上点的个数
        self._pt = None 
        self._els = []  #单元节点编号

    def init(self):
        x,y = np.meshgrid(self._X,self._Y) #转换为二位矩阵坐标
        self.x,self.y = x.flatten(),y.flatten() #将数组x,y的数据，均转换为一维向量
        self._pt = np.stack([self.x,self.y],axis=1) #将数组self.x,self.y堆叠

        ##这里是按照逆时针的顺序进行矩形单元节点编号
        for i in range(self._Ny):
            for j in  range(self._Nx):
                ind0 = i*(self._Nx + 1) + j
                ind1 = ind0 +1 
                ind2 = ind1 + (self._Nx + 1)
                ind3 = ind2 - 1
                self._els.append([ind0,ind1,ind2,ind3]) #得到矩形单位节点编号

    @property
    def points(self):   
        return self._pt  #矩形节点的坐标

    @property
    def els(self):
        return self._els #返回矩形单位节点编号


# In[51]:


class Node(object):
    def __init__(self,nid,x,y):
        self.ID = nid #nid单位节点个数
        self.x = x
        self.y = y
        self.nAk = ("Ux","Uy")
        ## 位移情况
        self.ux = None
        self.uy = None
        self.disp = dict.fromkeys(self.nAk,0.)
        ## 受力情况
        self.Fx,self.Fy = None,None


# In[52]:


class Rcent(object): #四边形应力单元
    
    def __init__(self,nel,nodes): #初始类别属性
        self.E = 2e5   #弹性模量
        self.nu = 0.3 #泊松比
        self.t = 0.025 #厚度
        self.nodes = nodes
        self.eIk = (("sx","sy","sxy"))
        self.k = self.calcRcent_Ke()
        self.stress =  dict.fromkeys(self.eIk,0.)
        self.strain =  dict.fromkeys(self.eIk,0.)
        self.u = None        
        
    def calcRcent_D(self): #计算本构矩阵D
        E = self.E
        nu = self.nu
        a = E/(1-nu**2)
        self.D = a*np.array([[1.,nu,0.],
                        [nu,1.,0.],
                        [0.,0.,(1-nu)/2.]])
        return self.D       
       
    def calcRcent_Ke(self):  #计算单元刚度矩阵，调用单元func(x)方法作为积分函数，并调用q1_quad2d求解二重积分
        self._Ke = gs2(self.Function,3)
        return self._Ke 
        
    def Function(self,x): #定义单元你的function(x)方法，为单元刚度矩阵函数
        self.calcRcent_D()
        nodes = self.nodes
        s = x[0]*1.0
        t = x[1]*1.0
        x1,y1 = nodes[0].x,nodes[0].y  #四边形节点坐标
        x2,y2 = nodes[1].x,nodes[1].y
        x3,y3 = nodes[2].x,nodes[2].y
        x4,y4 = nodes[3].x,nodes[3].y

        a = 1/4*(y1*(s-1)+y2*(-1-s)+y3*(1+s)+y4*(1-s))  #系数a,b,c,d
        b = 1/4*(y1*(t-1)+y2*(1-t)+y3*(1+t)+y4*(-1-t))
        c = 1/4*(x1*(t-1)+x2*(1-t)+x3*(1+t)+x4*(-1-t))
        d = 1/4*(x1*(s-1)+x2*(-1-s)+x3*(1+s)+x4*(1-s))

        B10 = -1/4*a*(1-t)+1/4*b*(1-s) #计算B1
        B11 = -1/4*c*(1-s)+1/4*d*(1-t)
        B20 = 1/4*a*(1-t)+1/4*b*(1+s) #计算B2
        B21 = -1/4*c*(1+s)-1/4*d*(1-t)
        B30 = 1/4*a*(1+t)-1/4*b*(1+s) #计算B3
        B31 = 1/4*c*(1+s)-1/4*d*(1+t)
        B40 = -1/4*a*(1+t)-1/4*b*(1-s) #计算B4
        B41 = 1/4*c*(1-s)+1/4*d*(1+t)
       
        B = np.array([[B10,   0, B20,   0, B30,   0, B40,  0],
                      [0,   B11, 0,   B21, 0,   B31, 0,  B41],
                      [B11, B10, B21, B20, B31 ,B30, B41,B40]])

        X = np.array([x1,x2,x3,x4])
        Y = np.array([y1,y2,y3,y4]).reshape(4,1)
        _J = np.array([[0,1-t,t-s,s-1],
                      [t-1,0,s+1,-s-t],
                      [s-t,-s-1,0,t+1],
                      [1-s,s+t,-t-1,0]])
        J = np.dot(np.dot(X,_J),Y)/8. #计算雅可比举证的行列式

        B = B/J
        J = J
        self.B = B 
        self.J = J
        return self.t*np.dot(np.dot(B.T,self.D),B)*J
    
    def assembelRcent_into_K(self,K):#组装整体刚度矩阵
        Ke = self.k
        nodes = self.nodes
        i = nodes[0].ID#单元的4个节点的ID：
        j = nodes[1].ID
        m = nodes[2].ID
        n = nodes[3].ID       
        K[2*i : 2*i+2 ,2*i :2*i+2] += Ke[0:2,0:2]
        K[2*i : 2*i+2, 2*j :2*j+2] += Ke[0:2,2:4]
        K[2*i : 2*i+2, 2*m: 2*m+2] += Ke[0:2,4:6]
        K[2*i : 2*i+2, 2*n: 2*n+2] += Ke[0:2,6:8]
        K[2*j :2*j+2, 2*i: 2*i+2] += Ke[2:4,0:2]
        K[2*j :2*j+2, 2*j: 2*j+2] += Ke[2:4,2:4]
        K[2*j: 2*j+2, 2*m: 2*m+2] += Ke[2:4,4:6]
        K[2*j: 2*j+2, 2*n: 2*n+2] += Ke[2:4,6:8] 
        K[2*m: 2*m+2, 2*i: 2*i+2] += Ke[4:6,0:2]
        K[2*m: 2*m+2, 2*j: 2*j+2] += Ke[4:6,2:4]
        K[2*m: 2*m+2, 2*m: 2*m+2] += Ke[4:6,4:6]
        K[2*m: 2*m+2, 2*n: 2*n+2] += Ke[4:6,6:8]
        K[2*n: 2*n+2, 2*i: 2*i+2] += Ke[6:8,0:2]
        K[2*n: 2*n+2, 2*j: 2*j+2] += Ke[6:8,2:4]
        K[2*n: 2*n+2, 2*m: 2*m+2] += Ke[6:8,4:6]
        K[2*n: 2*n+2, 2*n: 2*n+2] += Ke[6:8,6:8] 
                
#通过节点位移计算四边形单元应力与应变
    def evaluate(self):
        self.Function((0,0)) #用单元中心处的应力应变代表单元的应力应变
        u = np.array([[nd.disp[key] for nd in self.nodes for key in nd.nAk[:2]]])
        self._notdeal_stress = np.dot(np.dot(self.D,self.B),u.T)
        self._notdeal_strain = np.dot(self.B,u.T)
        n = len(self.eIk)
        for i,val in enumerate(self.eIk):
            self.stress[val] += self._notdeal_stress[i::n] #应力
            self.strain[val] += self._notdeal_strain[i::n] #应变
            
def gs2(fun,n,args=()): #高斯二重积分
        a,b= -1,1
        c,d = -1,1
        loc,w = np.polynomial.legendre.leggauss(n)
        s = (1/4.*(b-a)*(d-c)*fun(((b-a)*v1/2.+(a+b)/2.,
                               (d-c)*v2/2.+(c+d)/2.),*args)*w[i]*w[j]
             for i,v1 in enumerate(loc)
             for j,v2 in enumerate(loc))
        return sum(s)


# In[53]:


################################################### main-function(主函数) ###############################################################
size = 50
mesh = mesh([0,10],[0,10],size,size)
mesh.init()
els = mesh.els
ns = mesh.points
x = ns[:,0]
y = ns[:,1]

nodes = [  ] 
for v,nd in enumerate(ns):
    #函数enumerate()可以同时获得节点坐标的索引号和对应的坐标值，也就是说nd获得第v个节点的横纵坐标
    n = Node(v,x[v],y[v]) 
    nodes.append(n)  #创建节点
    
N = len(nodes)
nof = 2 # 节点自由度
K = np.zeros((N*2,N*2))
F = np.zeros(N*2)

q = 0  
#施加边界条件
for nd in nodes:
    if nd.x == 0.:
        nd.ux = 0.
        nd.uy = 0.
    if nd.x == 10  :
        F[2*nd.ID] = 1 # Fx作用在nd.ID上
        q = q+1

Els = [] #获得四边形单元
for v,el in enumerate(els):
    i,j,k,h = int(el[0]),int(el[1]),int(el[2]),int(el[3])
    n1,n2,n3,n4 = nodes[i],nodes[j],nodes[k],nodes[h]
    e = Rcent(v,(n1,n2,n3,n4))
    Els.append(e)  #创建四边形单元
p = 0            
for element in Els:
    element.assembelRcent_into_K(K)
    p = p +1


# In[54]:


print("划分网格数量为 :"+str(p) )


# In[55]:


# 生成U向量
# 注意位移的排列顺序
u = [[nd.ux,nd.uy] for nd in nodes]
U = np.array(u).flatten() #a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 


# In[56]:



keeped_ind = np.where(U == None)[0] # 划行划线法

keeped = [row for row,val in enumerate(U) if val is None]

keeped_K = K[keeped_ind,:][:,keeped_ind]
keeped_F = F[keeped_ind]


# In[57]:


keeped_U = np.linalg.inv(keeped_K).dot(keeped_F) #计算线性方程组
U[keeped_ind] = keeped_U # 所有节点的位移


# In[58]:


######################################################## 后处理 ##########################################################################
nAk = ("Ux","Uy")
for i,val in enumerate(keeped):
    I = val%2
    J = int(val/2)
    nodes[J].disp[nAk[I]] = keeped_U[i] #获取各个节点的相应位移
for el in Els:
    el.evaluate() #计算出四边形单元的应力、应变。


# In[59]:


U_x = U[0::2].tolist() #节点的x轴方向位移量
U_y = U[1::2].tolist()  #节点的y轴方向位移量


# In[60]:


U = []
for i in range(len(U_x)):
    U1 = U_x[i]**2 + U_y[i]**2
    U.append( U1**0.5)


# In[61]:


max(U)


# In[62]:


stress_x = np.array([el.stress["sx"][0][0] for el in Els])    #x轴方向上单元应力
stress_y = np.array([el.stress["sy"][0][0] for el in Els])    #y轴方向上单元应力
stress_xy = np.array([el.stress["sxy"][0][0] for el in Els])  #xy方向上的剪切应力
strain_x = np.array([el.strain["sx"][0][0] for el in Els])    #x轴方向上的单元应变
strain_y = np.array([el.strain["sy"][0][0] for el in Els])    #y轴方向上的单元应变
strain_xy = np.array([el.strain["sxy"][0][0] for el in Els])  #xy方向上的剪切应变

# In[64]:


max(stress_xy)


# In[65]:


fig2= plt.figure()
ax2 = fig2.add_subplot(111)
triangle = tri.Triangulation(x,y)
disp_ux = np.array(U_y,dtype=np.float)
t=ax2.tripcolor(triangle,disp_ux,color="k", cmap="jet",shading = 'gouraud')
ax2.set_title("python : U_y")
plt.colorbar(t)


# In[66]:


min(U_y)


# In[67]:


fig1= plt.figure()
ax1 = fig1.add_subplot(111)
triangle = tri.Triangulation(x,y)
disp_ux = np.array(U_x,dtype=np.float)
t=ax1.tripcolor(triangle,disp_ux,color="k", cmap="jet",shading = 'gouraud')
ax1.set_title("python : U_x")
plt.colorbar(t)


# In[68]:


max(U_x)


# In[69]:


fig3= plt.figure()
ax3 = fig3.add_subplot(111) 
patches = []    #x轴正应力图
#ex,ey = [],[]
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1)
pc.set_array(stress_x)
ax3.add_collection(pc)
ax3.set_xlim([0, 10])
ax3.set_ylim([0, 10])
ax3.set_aspect("equal")
ax3.set_title("python : stress_x")
fig3.colorbar(pc) 

fig4= plt.figure()
ax4 = fig4.add_subplot(111) 
patches = []    #y轴正应力图
#ex,ey = [],[]
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1)
pc.set_array(stress_y)
ax4.add_collection(pc)
ax4.set_xlim([0, 10])
ax4.set_ylim([0, 10])
ax4.set_aspect("equal")
ax4.set_title("python : stress_y")
fig4.colorbar(pc) 

fig5= plt.figure()
ax5 = fig5.add_subplot(111) 
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
ax5.add_collection(pc)
ax5.set_xlim([0, 10])
ax5.set_ylim([0, 10])
ax5.set_aspect("equal")
ax5.set_title("python : stress_xy")
fig5.colorbar(pc) 

fig6= plt.figure()
ax6 = fig6.add_subplot(111) 
patches = []    #x正应变图
#ex,ey = [],[]
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1)
pc.set_array(strain_x)
ax6.add_collection(pc)
ax6.set_xlim([0, 10])
ax6.set_ylim([0, 10])
ax6.set_aspect("equal")
ax6.set_title("python : strain_x")
fig6.colorbar(pc) 

fig7= plt.figure()
ax7 = fig7.add_subplot(111) 
patches = []    #y轴正应变图
#ex,ey = [],[]
for el in Els:
    ex, ey = [], []
    for nd in el.nodes:
        ex.append(nd.x)
        ey.append(nd.y)
    polygon = Polygon(list(zip(ex, ey)), True)
    patches.append(polygon)
pc = PatchCollection(patches,cmap="rainbow",color="k", alpha=1)
pc.set_array(strain_y)
ax7.add_collection(pc)
ax7.set_xlim([0, 10])
ax7.set_ylim([0, 10])
ax7.set_aspect("equal")
ax7.set_title("python : strain_y")
fig7.colorbar(pc) 

fig8= plt.figure()
ax8 = fig8.add_subplot(111) 
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
pc.set_array(strain_xy)
ax8.add_collection(pc)
ax8.set_xlim([0, 10])
ax8.set_ylim([0, 10])
ax8.set_aspect("equal")
ax8.set_title("python : strain_xy")
fig8.colorbar(pc) 
plt.show()

# In[71]:


max(strain_y)


# In[72]:


max(strain_x)


########"输出计算所得位移、应力应变等"##################
print("节点的x轴方向位移量\n")
print(U_x)
print("节点的y轴方向位移量\n")
print(U_y)
print("x轴方向上单元应力\n")
print(stress_x)
print("xy方向上的剪切应力\n")
print(stress_xy)
print("y轴方向上单元应力\n")
print(stress_y)
print("x轴方向上单元应变\n")
print(strain_x)
print("xy方向上剪切应变\%qtconsole")
print(strain_xy)
print("y轴方向上单元应变\n")
print(strain_y)


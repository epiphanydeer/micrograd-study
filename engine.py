# 有向无环图 DAG
'''
前向传播(Forward)：原料经过一道道工序（加法机、乘法机），加工出最终产品（Loss）。

反向传播（Backward）：质检员从最终产品出发，顺着流水线倒着追责，
告诉每个上游零件：“你对最终的残次/误差（Loss）要负多大责任（Gradient）”。
'''

class Value:
    # _children=() 用来追责定位父子节点
    # _op='' 用来记住运算操作
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        # 父节点没有_prev 不需要回传梯度
        self._backward = lambda : None
        self.grad = 0.0

    # 规范化输出
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Python 执行 a + b 时，等价于：c = a.__add__(b)
    # 当执行 c = a + b 时，发生三件事：
    # 计算纯数值：self.data + other.data（2.0 + 3.0 = 5.0）。
    # 记录血缘关系：把原料 (self, other) 即 (a, b) 打包成元组传给 _children。
    # 打上标签 '+'，并返回一个崭新的 Value 对象赋给 c。
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            # 为什么是累加？而不是赋值，因为当使用同一个变量的时候，会被重复的赋值为1，
            # 所以需要累加，以及为什么需要optimizer.zero_grad()上一个 batch 的梯度就会和当前 batch 叠在一起。
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other): # 如果2 * a不行，会尝试a * 2
        return self * other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            # 已知: out.grad = ∂L/∂out (全局导数)
            # a.grad = (∂out/∂a) * out.grad = b.data * out.grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(max(self.data, 0), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    '''
    调用 build_topo(L)：访问 L，发现 L 的原料是 y 深入 build_topo(y)。
    访问 y，发现 y 的原料是 x  深入 build_topo(x)。
    访问 x，x 没有原料（_prev 为空），递归到底，执行 topo.append(x)。
    回溯到 y，执行 topo.append(y)。回溯到 L，执行 topo.append(L)。
    递归结束得到的 topo 列表顺序为：[x, y, L]（严格的前向执行顺序：先有原料，再有半成品，后有成品）。
    执行 reversed(topo)：变成 [L, y, x]。
    这就是严格的反向传播执行顺序：先结算 L，再结算 y，最后结算 x。下游梯度的计算永远严格发生在所有上游之前。
    '''    
    def backward(self):

        topo = []
        visited = set()
        # 深度优先搜索 (DFS)
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1 # 起点：∂L/∂L = 1
        for v in reversed(topo):
            v._backward()
    '''
    Python 的 闭包机制, 节点身上背着的 c._backward，
    因为在定义计算算子的时候out._backward = _backward
    '''
    
a = Value(2.0)
b = Value(-3.0)
c = a * b
print(c._prev)
print(c._op)
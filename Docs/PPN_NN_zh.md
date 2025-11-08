# Implémentation d'un réseau de neurones pour la reconnaissance de chiffres manuscrits
手写数字识别神经网络的实现

Theoretical introduction
理论介绍

1. Statistical inference and the MNIST problem  
   统计推断与 MNIST 问题  
2. Introduction to neural networks  
   神经网络简介  
3. Computing the cost function's gradient with backpropagation  
   利用反向传播计算代价函数的梯度  
4. Generalization to automatic differentiation  
   推广到自动微分  
5. Gradient descent  
   梯度下降  
6. Steps for implementation
   实现步骤

# Statistical inference and the MNIST problem
统计推断与 MNIST 问题

Statistical inference is the process of inferring properties of a probability distribution by analysing some of its samples.
统计推断是通过分析概率分布的一些样本来推断其性质的过程。

- Nowadays, machine learning techniques are widely used to solve those problems, and especially neural networks.  
  如今，机器学习技术（尤其是神经网络）被广泛用于解决这些问题。  
These techniques typically work in two phases:
这些技术通常分两个阶段运行：

1. The training phase consists in analysing some given samples to understand their characteristics.  
   训练阶段通过分析给定样本来了解其特征。  
2. The prediction (or inference) phase consists in making predictions about new and unseen examples from what we have learned previously.
   预测（或推断）阶段利用已学习的内容对新的、未见过的样本进行预测。

- In this project, we are thus interested in a subset of predictive AI, which is to be differentiated from generative AI.
  因此，在本项目中我们关注的是预测式人工智能的一个子集，它需要与生成式人工智能区分开。

We'll start with a very simple linear regression example:
我们先从一个非常简单的线性回归例子开始：

- We want to predict a person's weight given their height...  
  我们希望根据一个人的身高预测其体重……  
... and assume the correlation between the two is linear.  
  ……并假设两者之间的相关性是线性的。  
- We will use a training dataset  $\{(x_i, y_i)\}_{i=1}^n$  to fine-tune a mathematical model for future predictions.
  我们将使用训练数据集 $\{(x_i, y_i)\}_{i=1}^n$ 来微调数学模型，以便进行未来的预测。

$$
\underbrace {y} _ {\text {p r e d i c t i o n}} = \underbrace {a} _ {\text {p a r a m e t e r}} \cdot \underbrace {x} _ {\text {i n p u t}} + \underbrace {b} _ {\text {p a r a m e t e r}} \tag {1}
$$

![](Images/PPN_image01.jpg)  
Figure 1: Linear regression on the height/weight example
图 1：身高与体重示例的线性回归

- To tune our model, we minimize the cost, defined as the sum of squared residuals:
  为了调整模型，我们最小化代价函数，即残差平方和：

$$
c = \sum_ {i = 1} ^ {n} \left(a. x _ {i} + b - y _ {i}\right) ^ {2} \tag {2}
$$

$\rightarrow$  This example is very simple and can be solved analytically using the least square method.
$\rightarrow$  这个例子非常简单，可以通过解析的最小二乘法求解。

# Introduction to the MNIST problem
MNIST 问题简介

- The MNIST dataset (Modified National Institute of Standards and Technology) consists of thousands of labeled  $28 \times 28$  grayscale images representing handwritten digits.  
  MNIST 数据集（修改版美国国家标准与技术研究院）包含成千上万个标注好的 $28 \times 28$ 手写数字灰度图像。  
- The goal of this project is to develop a program that classifies those digits in 10 categories. This will also consist in tuning a mathematical model, albeit more complex.
  本项目的目标是开发一个程序，将这些数字分成 10 个类别，这同样需要调节一个更复杂的数学模型。

![](Images/PPN_image02.jpg)  
Figure 2: Input (28x28 image) and output (column vector of size 10) of the MNIST problem
图 2：MNIST 问题的输入（28×28 图像）与输出（长度为 10 的列向量）

$$
y ^ {p r e d} = \left( \begin{array}{c} 1 \\ 0 \\ 0 \\ \vdots \\ 0 \end{array} \right)
$$

# Introduction to neural networks
神经网络简介

![](Images/PPN_image03.jpg)  
Figure 3: The Mark I Perceptron computer, an hardware implementation of the perceptron algorithm. (Source: Cornell University Library)
图 3：Mark I 感知机计算机，感知机算法的硬件实现。（来源：康奈尔大学图书馆）

We will solve MNIST using neural networks (NN), which are combinations of individual neurons.
我们将使用神经网络（NN）来解决 MNIST 问题，神经网络是多个单独神经元的组合。

- The concept of the multilayer perceptron, which is the NN model we are going to implement actually dates back to the 50s.  
  多层感知机这一概念（即我们将实现的神经网络模型）实际上可以追溯到 20 世纪 50 年代。  
- Back then, it was known as the perceptron algorithm, and was implemented in hardware by the Mark I Perceptron computer which was already used for image classification.
  当时它被称为感知机算法，并由 Mark I 感知机计算机以硬件形式实现，已经用于图像分类。

The concept of neurons takes its inspiration directly from biological neurons, where multiple output are combined to generate a single output:
神经元的概念直接来源于生物神经元，在那里多个输出被组合以产生单一输出：

![](Images/PPN_image04.jpg)  
Figure 4: Representation of a neuron, the elementary unit of neural networks
图 4：神经元示意图，神经网络的基本单元

Formally, a neuron is defined by the following function:
形式上，神经元可由以下函数定义：

$$
f: \left\{\begin{array}{l}\mathbb {R} ^ {n} \rightarrow [ 0, 1 ]\\x \mapsto \sigma \left(\sum_ {i = 1} ^ {n} x _ {i} w _ {i} + b\right)\end{array}\right. \tag {3}
$$

...which returns an activation value.
……它会返回一个激活值。

# Structure of a multilayer perceptron (MLP)
多层感知机（MLP）的结构

![](Images/PPN_image05.jpg)  
Figure 5: Structure of a multilayer perceptron, the most basic neural network architecture
图 5：多层感知机的结构，这是最基本的神经网络架构

- The multilayer perceptron is the simplest form of NN, made of one or multiple densely connected layers (all inputs connected to all outputs).  
  多层感知机是最简单的神经网络形式，由一个或多个稠密连接的层组成（所有输入都连接到所有输出）。  
Usually, consecutive layers become smaller and smaller as to "abstract" features of the original image.  
通常，后续的层会逐渐变小，以便对原始图像的特征进行“抽象”。  
- The size of the input layer is the size of the image, while the size of the output layer is the number of categories.
  输入层的大小等于图像的大小，而输出层的大小等于类别数量。

The value of a neuron  $i$  in a layer  $H$  is :
层 $H$ 中第 $i$ 个神经元的取值为：

$$
H _ {i} = \sigma \left(\sum_ {j = 1} ^ {I} I _ {j} w _ {j, i} + b _ {i}\right) \tag {4}
$$

Instead of considering each neuron individually, we notice that:
与其逐个考虑神经元，我们可以注意到：

$$
H = \sigma \left(\left[ \begin{array}{c c c c} w _ {1, 1} & w _ {1, 2} & \dots & w _ {1, l} \\ w _ {2, 1} & w _ {2, 2} & \dots & w _ {2, l} \\ \vdots & \vdots & \ddots & \vdots \\ w _ {m, 1} & w _ {m, 2} & \dots & w _ {m, l} \end{array} \right] \cdot \left[ \begin{array}{l} I _ {1} \\ I _ {2} \\ \vdots \\ I _ {l} \end{array} \right] + \left[ \begin{array}{l} b _ {1} \\ b _ {2} \\ \vdots \\ b _ {m} \end{array} \right]\right) \tag {5}
$$

As in the linear regression example, we try to minimize the model's cost (compared to the optimal solution  $\hat{y}$ ). It can first be defined as a vector:
与线性回归示例一样，我们尝试最小化模型的代价（相对于最优解 $\hat{y}$）。它可以首先表示为一个向量：

$$
\vec {c} = \left( \begin{array}{c} \left(\hat {y} _ {1} - y _ {1} ^ {\text {p r e d}}\right) ^ {2} \\ \left(\hat {y} _ {2} - y _ {2} ^ {\text {p r e d}}\right) ^ {2} \\ \vdots \\ \left(\hat {y} _ {n} - y _ {n} ^ {\text {p r e d}}\right) ^ {2} \end{array} \right) \tag {6}
$$

... and then as a scalar :
……随后也可以表示为一个标量：

$$
c = \sum_ {i = 1} ^ {n} c _ {i} = \sum_ {i = 1} ^ {n} \left(\hat {y} _ {i} - y _ {i} ^ {\text {p r e d}}\right) ^ {2} \tag {7}
$$

Computing each layer's output then the cost vector's norm yields the network's cost function.
计算每一层的输出，然后计算代价向量的范数，就得到网络的代价函数。

# Computing the cost function's gradient with backpropagation
利用反向传播计算代价函数的梯度

![](Images/PPN_image06.jpg)  
Figure 6: Intuitive representation of the gradient vector
图 6：梯度向量的直观表示

- As in the least square method, we'll use the cost function's gradient to search for a minimum.  
  与最小二乘法类似，我们会使用代价函数的梯度来寻找最小值。  
The gradient of a function  $f(x_{1},\ldots ,x_{n})$  is defined as:
函数 $f(x_{1},\ldots ,x_{n})$ 的梯度定义为：

$$
\nabla f = \left[ \begin{array}{c} \frac {\partial f}{\partial x _ {1}} \\ \frac {\partial f}{\partial x _ {2}} \\ \vdots \\ \frac {\partial f}{\partial x _ {n}} \end{array} \right] \tag {8}
$$

It is basically a vector that points us to a minimum of the cost function. Optimization is done by adjusting parameters according to it.
本质上这是一个指向代价函数最小值方向的向量，优化通过依据它来调整参数。

- Finding the expression of the cost function's partial derivatives is only possible with simple models.  
  只有对于简单模型，我们才能找到代价函数偏导数的解析表达式。  
- Because neural networks can become extremely complex (easily thousands of parameters), we can hardly express the partial derivatives through symbolic differentiation.  
  因为神经网络可能非常复杂（动辄包含成千上万的参数），几乎无法通过符号求导写出偏导。  
- We will give up on finding the gradient's expression and compute it on the fly for each point we examine in a step called backpropagation.
  因此我们不再追求梯度的解析形式，而是在名为反向传播的步骤中对每个点即时计算梯度。

The backpropagation (or backward pass) does not express a gradient, but only computes it for a given point.
反向传播（或称反向传递）不会给出梯度的解析表达式，只会针对给定点计算梯度。

It relies on the chain rule :
它依赖于链式法则：

$$
(f \circ g) ^ {\prime} = \left(f ^ {\prime} \circ g\right). g ^ {\prime} \tag {9}
$$

which can be written as:
其可以写为：

$$
\frac {\mathrm {d} \mathbf {z}}{\mathrm {d} \mathbf {x}} = \frac {\mathrm {d} \mathbf {z}}{\mathrm {d} \mathbf {y}} \cdot \frac {\mathrm {d} \mathbf {y}}{\mathrm {d} \mathbf {x}} \tag {10}
$$

...where  $y = g(x)$  and  $z = f(y) = f(g(x))$ .
……其中 $y = g(x)$，$z = f(y) = f(g(x))$。

The idea is to express a complex function as a combination of "elementary operations". An elementary operation is any function for which partial derivatives are known, such as:
其思想是将复杂函数表示为“基本运算”的组合。基本运算指偏导数已知的函数，例如：

- additions +, subtractions -, multiplications  $\times$ , divisions  $\div$  
  加法 +、减法 -、乘法 $\times$、除法 $\div$  
- trigonometric functions such as  $\cos, \sin, \tan, \text{etc.}$  
  三角函数，如 $\cos, \sin, \tan$ 等  
and any other "trivial" functions (e, log,...)
以及任何其他“简单”的函数（如 $e$、$\log$ 等）

Such a decomposition yields a computation graph. Using the chain rule, it is possible to compute the partial derivatives of each intermediate node until the end of the graph, which will give us the partial derivatives (ie. the gradient) of the entire function!
这种分解会带来一张计算图。利用链式法则，可以一直计算到图的末尾，从而得到整个函数的偏导数（即梯度）！

Problem: if an elementary operation has multiple inputs, they all need to be accounted for when computing its partial derivative with regard to  $x$  using the chain rule. This is because all those inputs may depend on  $x$  themselves.
问题：如果一个基本运算有多个输入，在使用链式法则计算关于 $x$ 的偏导时，需要把所有输入都考虑进去，因为它们本身可能依赖于 $x$。

This leads to the following generalization:
这就引出了如下推广形式：

$$
\frac {\partial x}{\partial z} = \sum_ {i = 1} ^ {n} \frac {\partial y _ {i}}{\partial z} \cdot \frac {\partial x}{\partial y _ {i}} \tag {11}
$$

...where  $z$  depends on a set  $\{y_{i}\}_{i = 1}^{n}$  of variables, each of them depending on  $x$ .
……其中 $z$ 依赖于一组变量 $\{y_{i}\}_{i = 1}^{n}$，每个变量都依赖于 $x$。

Let us first introduce gradient computation in forward mode. We have the function
我们先介绍正向模式的梯度计算。我们有如下函数

$$
f: \left\{\begin{array}{l}\mathbb {R} ^ {2} \rightarrow \mathbb {R}\\\left(x _ {1}, x _ {2}\right) \mapsto x _ {2}. c o s \left(x _ {1}\right) + e ^ {x _ {2}}\end{array}\right. \tag {12}
$$

With denoting partial derivatives, the computation graph is:
在标注偏导数后，计算图如下：

![](Images/PPN_image07.jpg)  
Figure 7: Base computation graph of  $f$
图 7：函数 $f$ 的基础计算图

The partial derivatives will be computed left-to-right by resolving the nodes' dependencies:
偏导数将通过按依赖关系自左向右地解析节点来计算：

![](Images/PPN_image08.jpg)  
Figure 8: Detailed chain rule in forward mode for  $f$ , with regard to  $x_{1}$
图 8：函数 $f$ 在 $x_{1}$ 方向上的正向链式法则细节

The partial derivatives will be computed left-to-right by resolving the nodes' dependencies:
偏导数将通过按依赖关系自左向右地解析节点来计算：

![](Images/PPN_image09.jpg)  
Figure 8: Detailed chain rule in forward mode for  $f$ , with regard to  $x_{1}$
图 8：函数 $f$ 在 $x_{1}$ 方向上的正向链式法则细节

The partial derivatives will be computed left-to-right by resolving the nodes' dependencies:
偏导数将通过按依赖关系自左向右地解析节点来计算：

![](Images/PPN_image10.jpg)  
Figure 8: Detailed chain rule in forward mode for  $f$ , with regard to  $x_{1}$
图 8：函数 $f$ 在 $x_{1}$ 方向上的正向链式法则细节

The partial derivatives will be computed left-to-right by resolving the nodes' dependencies:
偏导数将通过按依赖关系自左向右地解析节点来计算：

![](Images/PPN_image11.jpg)  
Figure 8: Detailed chain rule in forward mode for  $f$ , with regard to  $x_{1}$
图 8：函数 $f$ 在 $x_{1}$ 方向上的正向链式法则细节

$$
\left\{ \begin{array}{l} f (\pi , 1) = 1. \cos (\pi) + e ^ {1} = e - 1 \\ \frac {\partial f}{\partial x _ {1}} = - 1. \sin (\pi) = 0 \\ \frac {\partial f}{\partial x _ {2}} = \cos (\pi) + e ^ {1} = e - 1 \end{array} \right. \tag {13}
$$

Table 1: Step-by-step computation of  $\partial f$  in forward mode  
表 1：正向模式下逐步计算 $\partial f$  

<table><tr><td>Evaluation of f</td><td>Evaluation of ∂f/∂x1</td><td>Evaluation of ∂f/∂x2</td></tr><tr><td>v-1=x1=π</td><td>v&#x27;-1=x1=1</td><td>v&#x27;-1=x1=0</td></tr><tr><td>v0=x2=1</td><td>v0=x2=0</td><td>v0=x2=1</td></tr><tr><td>v1=cos(v-1)=-1</td><td>v1=-v&#x27;-1.sin(v-1)=0</td><td>v1=-v&#x27;-1.sin(v-1)=0</td></tr><tr><td>v2=v0.v1=-1</td><td>v2=v0.v1+v0.v1=0</td><td>v2=v0.v1+v0.v1=-1</td></tr><tr><td>v3=eV0=e</td><td>v3=v0.eV0=0</td><td>v3=v0.eV0=e</td></tr><tr><td>v4=v2+v3=e-1</td><td>v4=v2+v3=0</td><td>v4=v2+v3=e-1</td></tr></table>

We will now introduce backward mode. The principle is the same, but we rewrite the chain rule by inverting its inputs and outputs. Equation (10)
接下来我们介绍反向模式。原理相同，但通过交换链式法则的输入与输出来改写。式（10）

$$
\frac {d z}{d x} = \frac {d z}{d y} \cdot \frac {d y}{d x}
$$

becomes:
变为：

$$
\frac {\mathrm {d} \mathbf {x}}{\mathrm {d} \mathbf {z}} = \frac {\mathrm {d} \mathbf {x}}{\mathrm {d} \mathbf {y}} \cdot \frac {\mathrm {d} \mathbf {y}}{\mathrm {d} \mathbf {z}} \tag {14}
$$

... where  $\frac{dx}{dz}$  and  $\frac{dy}{dz}$  are called the adjoints. Using this, we can compute partial derivatives by starting at the end of the graph and taking  $\frac{dz}{dz} = 1$ . Similarly, the generalization to multiple variables (when an output is reused as input multiple times) is:
……其中 $\frac{dx}{dz}$ 和 $\frac{dy}{dz}$ 被称为伴随量。借此我们可以从图的末端开始计算偏导，并令 $\frac{dz}{dz} = 1$。当一个输出被多次用作输入时，其多变量推广形式为：

$$
\frac {\partial x}{\partial z} = \sum_ {i = 1} ^ {n} \frac {\partial x}{\partial y _ {i}} \cdot \frac {\partial y _ {i}}{\partial z} \tag {15}
$$

After computing the function and the intermediate partial derivatives by going through the graph normally, we will go backward from the last node, with  $\bar{\cdot}$  denoting the adjoints:
在按常规顺序计算完函数及中间偏导之后，我们将从最后一个节点反向遍历，其中 $\bar{\cdot}$ 表示相应的伴随量：

![](Images/PPN_image12.jpg)  
Figure 9: Backward computation graph of  $\partial f$
图 9：$\partial f$ 的反向计算图

In this backward pass, we apply the reversed chain rule to get the adjoints.
在这次反向传播中，我们应用反向的链式法则来获取伴随量。

Graphically, and after computing the partial derivatives during the forward pass:
从图形上看，在前向传播中计算完偏导之后：

![](Images/PPN_image13.jpg)  
Figure 10: Detailed chain rule in backward mode for  $f$
图 10：函数 $f$ 在反向模式下的链式法则细节

Graphically, and after computing the partial derivatives during the forward pass:
从图形上看，在前向传播中计算完偏导之后：

![](Images/PPN_image14.jpg)  
Figure 10: Detailed chain rule in backward mode for  $f$
图 10：函数 $f$ 在反向模式下的链式法则细节

Graphically, and after computing the partial derivatives during the forward pass:
从图形上看，在前向传播中计算完偏导之后：

![](Images/PPN_image15.jpg)  
Figure 10: Detailed chain rule in backward mode for  $f$
图 10：函数 $f$ 在反向模式下的链式法则细节

Graphically, and after computing the partial derivatives during the forward pass:
从图形上看，在前向传播中计算完偏导之后：

![](Images/PPN_image16.jpg)  
Figure 10: Detailed chain rule in backward mode for  $f$
图 10：函数 $f$ 在反向模式下的链式法则细节

Table 2: Step-by-step computation of  $\partial f$  in backward mode  
表 2：反向模式下逐步计算 $\partial f$  

<table><tr><td>Evaluation of f</td><td>Evaluation of the ∂f/∂xi</td></tr><tr><td>v-1=x1=π</td><td>v̅_1 = v̅_1. ∂v_1/∂v_1 = 0</td></tr><tr><td>v0=x2=1</td><td>v̅_0 = v̅_3. ∂v_3/∂v_0 + v̅_2. ∂v_2/∂v_0 = e - 1</td></tr><tr><td>v1=cos(v-1)=-1</td><td>v̅_1 = v̅_2. ∂v_2/∂v_1 = -1</td></tr><tr><td>v2=v0.v_1=-1</td><td>v̅_2 = v̅_4. ∂v_4/∂v_2 = -1</td></tr><tr><td>v_3=e^{v_0}=e</td><td>v̅_3 = v̅_4. ∂v_4/∂v_3 = e</td></tr><tr><td>v_4=v_2+v_3=e-1</td><td>v̅_4 = 1</td></tr></table>

Notice how this only required a single backward pass (instead of two in forward mode)!
请注意，这里只需要一次反向传播（而正向模式需要两次）！

In the general case of a function  $f: \mathbb{R}^m \to \mathbb{R}^n$ , its Jacobian matrix is given by:
在一般情况下，对于函数 $f: \mathbb{R}^m \to \mathbb{R}^n$，其雅可比矩阵如下：

$$
J _ {f} = \left[ \begin{array}{c c c c} \frac {\partial f _ {1}}{\partial x _ {1}} & \frac {\partial f _ {1}}{\partial x _ {2}} & \dots & \frac {\partial f _ {1}}{\partial x _ {m}} \\ \frac {\partial f _ {2}}{\partial x _ {1}} & \frac {\partial f _ {2}}{\partial x _ {2}} & \dots & \frac {\partial f _ {2}}{\partial x _ {m}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac {\partial f _ {n}}{\partial x _ {1}} & \frac {\partial f _ {n}}{\partial x _ {2}} & \dots & \frac {\partial f _ {n}}{\partial x _ {m}} \end{array} \right] \tag {16}
$$

Notice how forward mode computes the Jacobian column by column, and backward mode row by row.
可以看到，正向模式按列计算雅可比矩阵，而反向模式按行计算。

- if  $n > m$ , it's easier to compute  $J_{f}$  in forward mode ( $m$  passes)  
  如果 $n > m$，使用正向模式计算 $J_{f}$ 更容易（需要 $m$ 次传递）。  
- if  $n < m$ , it's easier to compute  $J_{f}$  in backward mode ( $n$  passes)
  如果 $n < m$，使用反向模式计算 $J_{f}$ 更容易（需要 $n$ 次传递）。

$\rightarrow$  In our context, only backward mode is relevant because our cost functions have a very large number of inputs for only a single output ( $n = 1$ , the Jacobian has a single row).
$\rightarrow$  在我们的场景中只有反向模式才重要，因为代价函数拥有大量输入却只有一个输出（$n = 1$，雅可比矩阵只有一行）。

# Generalization to automatic differentiation
推广到自动微分

# Autodiff implementation using a gradient tape
使用梯度带实现自动微分

The principle of a backward mode autodiff implementation is to store the computation graph in a gradient tape (sometimes called a Wengert list, hence the  $W$  notation) which is built during the forward pass:
反向模式自动微分实现的原理，是在前向传播过程中把计算图存入梯度带（有时称为 Wengert 表，因此使用 $W$ 标记）。

![](Images/PPN_image17.jpg)  
Figure 11: Gradient tape of  $f$
图 11：函数 $f$ 的梯度带

Once the tape is fully built, each node contains:
梯度带构建完成后，每个节点都包含：

- the partial derivatives of its variable with regard to their parent variables  
  该变量相对于其父节点变量的偏导数  
- the indices of its parent nodes
  父节点的索引

The gradient tape doesn't need to contain the values of the intermediate variables, only the partial derivatives.
梯度带不需要保存中间变量的取值，只需存储偏导数。

Computing a gradient from a gradient tape is done using the following algorithm:
从梯度带中计算梯度可以通过以下算法完成：

Algorithm 1: Gradient computation from a gradient tape in backward mode  
算法 1：基于梯度带的反向模式梯度计算  
Input: gradient tape  $W$  of size  $k$ , variable  $\nu$  associated to the node  $W_{k}$   
输入：大小为 $k$ 的梯度带 $W$，变量 $\nu$ 对应节点 $W_{k}$  
Result: gradient  $\nabla$   
输出：梯度 $\nabla$  
1  $\nabla \gets \{0\}_{0}^{|W|}$   
1  将 $\nabla$ 初始化为 $\{0\}_{0}^{|W|}$  
2  $\nabla_{k} \gets 1$   
2  将 $\nabla_{k}$ 设为 1  
/* Backward pass */  
/* 反向遍历 */  
3 for  $i \gets k$  to 1 do  
3 对 $i$ 从 $k$ 递减到 1 执行循环  
4 P ← parents(Wi)  
4 令 P ← parents(Wi)  
for  $j \gets 0$  to  $|P|$  do  
对 $j$ 从 0 到 $|P|$ 进行循环  
/* Index in  $P$  != Index in  $W$  or  $\nabla$  */  
/* P 中的索引与 W 或 $\nabla$ 中的索引不同 */  
j' ← index $_{\nabla}(P_j)$ $\nabla_{j'} \gets \nabla_{j'} + \frac{\partial C_i}{\partial W_k} \cdot \frac{\partial C_j}{\partial C_i}$   
j' ← index$_{\nabla}(P_j)$，将 $\nabla_{j'}$ 增加 $\frac{\partial C_i}{\partial W_k} \cdot \frac{\partial C_j}{\partial C_i}$  
end  
结束内部循环  
9 end
9 结束外层循环

# Autodiff implementation using a gradient tape
使用梯度带实现自动微分

An elegant way of implementing this is through operator overloading. This approach has multiple advantages:
一种优雅的实现方式是通过运算符重载。这种方法有多重优势：

- Almost transparent  
  几乎是透明的  
Gradient can simply be retrieved in a single function call  
梯度可以通过一次函数调用直接获取  
Efficient
效率高

Remark: About tensor generalization...
备注：关于张量的推广……

- In a "realistic" neural network, the gradient tape would grow huge if we only considered scalar nodes!  
  在“真实”的神经网络中，如果只考虑标量节点，梯度带会变得非常庞大！  
- To fix this, we can work directly on tensor nodes instead.  
  为了解决这个问题，我们可以直接处理张量节点。  
- This makes it possible to write a matrix product in a single node, for instance!  
  例如，这样一来可以在单个节点中表示矩阵乘法！  
However, it is OK to focus on scalars for a first implementation.
不过，在首次实现时专注于标量已经足够。

# Gradient descent
梯度下降

The  $-\nabla$  hints us to the direction of a minimum of the cost function:
$-\nabla$ 指示了代价函数最小值所在的方向：

![](Images/PPN_image18.jpg)  
Figure 12: Gradient descent example
图 12：梯度下降示例

By adjusting the model's parameters in small steps (ie. the weights and biases), we converge to a minimum of the cost function and our predictions become better and better!
通过以很小的步长（即权重和偏置）调整模型参数，我们可以收敛到代价函数的一个最小值，从而不断提升预测效果！

The SGD (stochastic gradient descent) algorithm is the most basic optimization algorithm. For each training sample:
SGD（随机梯度下降）算法是最基本的优化算法。对于每一个训练样本：

- We define a learning rate  $\eta$  (usually very small). This allows us to take small steps at a time and converge precisely.  
  我们设定一个学习率 $\eta$（通常很小），让每一步都很小以便精确收敛。  
We compute  $\nabla$  on some training data  
我们在部分训练数据上计算 $\nabla$  
We increment the model by  $-\eta \nabla$
然后按 $-\eta \nabla$ 更新模型

After hundreds or thousands of iterations, the cost function should converge!
经过上百甚至上千次迭代后，代价函数应当会收敛！

The basic SGD algorithm is the following:
基本的 SGD 算法如下：

Algorithm 2: Stochastic gradient descent (SGD) algorithm
算法 2：随机梯度下降（SGD）算法

Input: function  $f$ , initial parameters  $\theta$ , set of training updates  $U$ , epoch numbers  $e$ , learning rate  $\eta$
输入：函数 $f$、初始参数 $\theta$、训练更新集合 $U$、迭代轮数 $e$、学习率 $\eta$

Result: optimized parameters  $\theta$
输出：优化后的参数 $\theta$

1 for  $i\gets 1$  to e do   
1 对 $i$ 从 1 到 $e$ 进行循环  
2 for  $j\gets 1$  to  $|U|$  do   
2 对 $j$ 从 1 到 $|U|$ 进行循环  
3  $\begin{array}{c}b\gets B_j\\ \nabla \leftarrow \nabla f_\theta (u_j)\\ \theta \leftarrow \theta -\eta \nabla \end{array}$    
3  令 $b \gets B_j$，计算 $\nabla \leftarrow \nabla f_\theta (u_j)$，并更新 $\theta \leftarrow \theta - \eta \nabla$    
6 end   
6 结束内层循环   
7 end
7 结束外层循环

Problem: Updating the model after each training step makes it subject to overfitting, which can degrade the prediction's precision.
问题：每次训练都立即更新模型会导致过拟合，从而降低预测精度。

# The problem of overfitting
# 过拟合问题

Overfitting happens when a model has been trained to correspond too closely to a particular set of data, making it less general and less efficient at making predictions on new, unseen data.
当模型被训练得过于贴合特定数据集时，就会发生过拟合，导致泛化能力下降，对新的未见数据的预测效果变差。

![](Images/PPN_image19.jpg)  
Figure 13: Example of overfitting against the linear regression example: the model fits each training instance too closely.
图 13：在线性回归示例中发生过拟合的情况：模型过度拟合每个训练样本。

Remark: Overfitting is also more likely when using a model with too many parameters.
备注：当模型参数过多时，更容易出现过拟合。

To solve this issue, we often group the training inputs in small batches, and update the model only once per batch:
为了解决这个问题，我们通常将训练输入分成小批次，并且每个批次只更新一次模型：

Input: function  $f$ , initial parameters  $\theta$ , set of batches  $B$ , epoch numbers  $e$ , learning rate  $\eta$
输入：函数 $f$、初始参数 $\theta$、批次集合 $B$、迭代轮数 $e$、学习率 $\eta$

Result: optimized parameters  $\theta$
输出：优化后的参数 $\theta$

Algorithm 3: Batched SGD algorithm  
算法 3：批量 SGD 算法  
1 for  $i\gets 1$  to e do  
1 对 $i$ 从 1 到 $e$ 进行循环  
10 end  
10 结束循环  
```latex
for  $j\gets 1$  to  $|B|$  do   
3  $\begin{array}{l}b\gets B_j\\ \nabla_{acc}\gets \{0\}_{k = 1}^{|\theta |}\\ \text{for} k\gets 1\text{to}|b|\text{do}\\ |\nabla_{acc}\gets \nabla_{acc} + \nabla f_\theta (b_k)\\ \text{end}\\ \theta \leftarrow \theta -\eta \nabla_{acc} \end{array}$    
4   
5   
6   
7   
8   
9
```

The variable  $\nabla_{acc}$  is called a gradient accumulator.
变量 $\nabla_{acc}$ 被称为梯度累加器。

# Steps for implementation
# 实现步骤

Once you are familiar with all the theory, here are the different steps you should follow during the first semester:
当你熟悉了所有理论之后，以下是第一学期应当完成的各个步骤：

- Implement a simple MLP (forward pass only for now).  
  实现一个简单的 MLP（目前只需前向传播）。  
- Implement a simple autodiff engine working in backward mode to generate the gradient tape.  
  实现一个在反向模式下工作的简单自动微分引擎，用于生成梯度带。  
- Make sure it supports matrix/vector nodes. You will need to implement the matrix (or matrix-vector) product, the element-wise sums and products, and an element-wise activation function.
  确保它支持矩阵/向量节点。你需要实现矩阵（或矩阵-向量）乘法、逐元素加法与乘法，以及逐元素激活函数。

- This includes being able to compute the partial derivatives of those operations.
  这也意味着需要能够计算这些运算的偏导数。

- Combine the two, as to be able to compute the MLP's gradient  
  将两者结合起来，从而能够计算 MLP 的梯度。  
- Implement the SGD algorithm, activation function, weights and biases initialization, ...  
  实现 SGD 算法、激活函数、权重和偏置的初始化等。  
- Combine all of this on the MNIST dataset (you will be given some code to read the data files so you can focus on the most important part of the project).
  在 MNIST 数据集上整合以上所有内容（你将获得读取数据文件的代码，以便专注于项目最重要的部分）。

After all that, you will have implemented a functional MNIST solver! During the second semester, the goal will be to improve its performance and accuracy.
完成以上所有步骤后，你就会实现一个可用的 MNIST 求解器！在第二学期，目标将是提升其性能和准确率。
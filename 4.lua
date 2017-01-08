
require 'nn' -- package for neural networks
Plot = require 'itorch.Plot' -- package for creating plots

function hardlim(result)
  newResult = torch.Tensor(1,2)

  if(result[1][1] > 0) then
    newResult[1][1] = 1
  else
    newResult[1][1] = 0
  end

  if(result[1][2] > 0) then
    newResult[1][2] = 1
  else
    newResult[1][2] = 0
  end

  return newResult
end

net = nn.Linear(2,2) -- create perceptron with 2-neuron output layer

weight = torch.Tensor(2,2)
weight[1][1] = 0.8
weight[1][2] = -0.2
weight[2][1] = -0.2
weight[2][2] = 0.8

net.weight = weight

bias = torch.Tensor(2)
bias[1] = 1
bias[2] = -1
net.bias = bias

print('first neuron weights:', net.weight[1][1], net.weight[1][2]) -- print weights
print('first neuron bias:',net.bias[1]) -- print biases
print('-----------------------')
print('second neuron weights:', net.weight[2][1], net.weight[2][2]) -- print weights
print('second neuron bias:',net.bias[2]) -- print biases
print('-----------------------')
testX1 = 0
testY1 = 0
print('First test point:', testX1, testY1)

p1 = torch.Tensor(1,2)
p1[1][1] = testX1
p1[1][2] = testY1
result1 = hardlim(net:forward(p1))
print('result1:', result1[1][1], result1[1][2])
print('-----------------------')

testX2 = 2.5
testY2 = -1
print('Second test point:', testX2, testY2)

p2 = torch.Tensor(1,2)
p2[1][1] = testX2
p2[1][2] = testY2
result2 = hardlim(net:forward(p2))
print('result2:', result2[1][1], result2[1][2])
print('------------------------')

testX3 = -3
testY3 = -3
print('Third test point:', testX3, testY3)

p3 = torch.Tensor(1,2)
p3[1][1] = testX3
p3[1][2] = testY3
result3 = hardlim(net:forward(p3))
print('result3:', result3[1][1], result3[1][2])
print('-------------------------')

testX4 = 4
testY4 = 4
print('Fourth test point:', testX4, testY4)

p4 = torch.Tensor(1,2)
p4[1][1] = testX4
p4[1][2] = testY4
result4 = hardlim(net:forward(p4))
print('result4:', result4[1][1], result4[1][2])
print('--------------------------')

testX5 = -4
testY5 = 3
print('Fiveth test point:', testX5, testY5)

p5 = torch.Tensor(1,2)
p5[1][1] = testX5
p5[1][2] = testY5
result5 = hardlim(net:forward(p5))
print('result5:', result5[1][1], result5[1][2])

x = torch.linspace(-5,5)
y1 = -(net.bias[1] + net.weight[1][1]*x)/net.weight[1][2] -- creating first line
y2 = -(net.bias[2] + net.weight[2][1]*x)/net.weight[2][2] -- creating second line

plot = Plot():line(x, y1,'red','first neuron'):line(x,y2,'green','second neuron'):legend(true):title(''):draw()
plot:circle(torch.Tensor({testX1}),torch.Tensor({testY1}),'blue','test point 1'):redraw()
plot:circle(torch.Tensor({testX2}),torch.Tensor({testY2}),'yellow','test point 2'):redraw()
plot:circle(torch.Tensor({testX3}),torch.Tensor({testY3}),'black','test point 3'):redraw()
plot:circle(torch.Tensor({testX4}),torch.Tensor({testY4}),'orange','test point 4'):redraw()
plot:circle(torch.Tensor({testX5}),torch.Tensor({testY5}),'purple','test point 5'):redraw()

plot:save('plots.html')

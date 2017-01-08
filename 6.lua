require 'nn'

function hardlim(input)
  if(input[1][1] > 0) then
    return 1
  else 
    return 0
  end
end

function learnp(p,e)
  return e*p
end

net = nn.Linear(2,1)

net.weight = torch.Tensor(1,2):fill(1)
net.bias = torch.Tensor(1):zero()

print('First vector:')
p1 = torch.Tensor({{-2,1}})
t1 = 0
result1 = hardlim(net:forward(p1))
print('expected:',t1)
print('got:',result1)
e = t1 - result1
dw = learnp(p1,e)
net.weight = net.weight + dw
print('new weights:',net.weight[1][1],net.weight[1][2])
print('----------------------------------')

print('Second vector:')
p2 = torch.Tensor({{0,1}})
t2 = 0
result2 = hardlim(net:forward(p2))
print('expected:',t2)
print('got:',result2)
e = t2 - result2
dw = learnp(p2,e)
net.weight = net.weight + dw
print('new weights:',net.weight[1][1],net.weight[1][2])
print('------------------------------------')

print('Third vector:')
p3 = torch.Tensor({{2,-1}})
t3 = 1
result3 = hardlim(net:forward(p3))
print('expected:',t3)
print('got:',result3)
e = t3 - result3
dw = learnp(p3,e)
net.weight = net.weight + dw
print('new weights:',net.weight[1][1],net.weight[1][2])
print('------------------------------------')

print('Fourth vector:')
p4 = torch.Tensor({{-2,1}})
t4 = 0
result4 = hardlim(net:forward(p4))
print('expected:',t4)
print('got:',result4)
e = t4 - result4
dw = learnp(p4,e)
net.weight = net.weight + dw
print('new weights:',net.weight[1][1],net.weight[1][2])
print('--------------------------------------')

print('First vector again with new weigts:')
result1 = hardlim(net:forward(p1))
print('expected:',t1)
print('got:',result1)
e = t1 - result1
dw = learnp(p1,e)
net.weight = net.weight + dw
print('new weights:',net.weight[1][1],net.weight[1][2])

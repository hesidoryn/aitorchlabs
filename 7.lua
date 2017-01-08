require 'nn'

function hardlim(input)
  if(input[1] > 0) then
    return 1
  else
    return 0
  end
end

function learn(net,p,e)
  net.weight = net.weight + p*e
  net.bias = net.bias + e
end

function test(net,p)
  size = p:size()[1]
  for i = 1, size do
    result = hardlim(net:forward(p[i]))
    print(result)
  end
end

function adapt(net,p,t)
  size = p:size()[1]

  for i = 1, size do
    result = hardlim(net:forward(p[i]))
    e = t[i] - result
    learn(net,p[i],e)
  end  
end

net = nn.Linear(2,1)

net.weight = torch.Tensor(1,2):fill(1)
net.bias = torch.Tensor(1):zero()

p = torch.Tensor(4,2)
p[1][1] = -2
p[1][2] = 1
p[2][1] = 0
p[2][2] = 1
p[3][1] = 2
p[3][2] = -1
p[4][1] = -2
p[4][2] = -1

t = torch.Tensor(4):zero()
t[3] = 1

print('Output before learning:')
test(net,p)

print('learning started...')
adapt(net,p,t)
print('learning finished')
print('-------------------')
print('Updated weigth and bias:')
print('weights:',net.weight[1][1],net.weight[1][2])
print('bias:',net.bias[1])

print('---------------------')
print('Output after learning:')
test(net,p)

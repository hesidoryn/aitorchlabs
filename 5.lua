require 'nn'
Plot = require 'itorch.Plot'

function tribas(input)
  size = input:size()[1]
  output = torch.Tensor(size)

  for i = 1, size do
    if(input[i] >= -1 and input[i] <= 1) then
      output[i] = 1 - torch.abs(input[i])
    else
      output[i] =  0
    end
  end

  return output
end

function purelin(input)
  output = input:clone()

  return output
end

function hardlim(input)
  size = input:size()[1]
  output = torch.Tensor(size)

  for i = 1, size do
    if(input[i] > 0) then
      output[i] = 1
    else
      output[i] = 0
    end
  end

  return output
end

x = torch.linspace(-5,5)
y1 = tribas(x) -- tribas function result

logsig = nn.Sigmoid()
y2 = logsig:forward(x) -- logsig function result

y3 = hardlim(x) -- hardlim function result

y4 = purelin(x) -- purelin function result

plot = Plot():line(x, y1,'red','tribas'):legend(true):title('Second lab: tribas function\'s plot'):draw()
plot:save('second1.html')

plot = Plot():line(x, y1,'red','tribas'):line(x,y2,'green','logsig'):line(x,y3,'yellow','hardlim'):line(x,y4,'blue','purelin'):legend(true):title('Second lab: all functions plots'):draw()
plot:save('second2.html')

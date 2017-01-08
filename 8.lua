cherry = torch.Tensor({0,0,1,1,0,1,0})
orange = torch.Tensor({0,1,0,0,1,0,0})
apple = torch.Tensor({1,0,1,0,1,0,0})
melon = torch.Tensor({1,0,0,0,1,0,1})
banana = torch.Tensor({1,0,0,0,0,0,0})

function calculateA(x1, x2) 
    result = 0
    for i = 1, x1:size(1) do
        result = result + x1[i] * x2[i]
    end
    return result
end

function calculateB(x1, x2)
    result = 0
    for i = 1, x1:size(1) do
        result = result + (1 - x1[i]) * (1 - x2[i])
    end
    return result
end

function calculateG(x1, x2)
    result = 0
    for i = 1, x1:size(1) do
        result = result + (1 - x1[i]) * x2[i]
    end
    return result
end

function calculateH(x1, x2)
    result = 0
    for i = 1, x1:size(1) do
        result = result + x1[i] * (1 - x2[i])
    end
    return result
end

function S1(x1,x2)
    a = calculateA(x1,x2)

    return a/x1:size(1)
end

function S2(x1,x2)
    a = calculateA(x1,x2)
    b = calculateB(x1,x2)

    return a/(x1:size(1) - b)
end

function S3(x1,x2)
    a = calculateA(x1,x2)
    g = calculateG(x1,x2)
    h = calculateH(x1,x2)

    return a/(2 * a + g + h)
end

function S4(x1,x2)
    a = calculateA(x1,x2)
    g = calculateG(x1,x2)
    h = calculateH(x1,x2)

    return a/(a + 2 * (g + h))
end

function S5(x1,x2)
    a = calculateA(x1,x2)
    b = calculateB(x1,x2)

    return (a + b)/x1:size(1)
end

function S6(x1,x2)
    a = calculateA(x1,x2)
    b = calculateB(x1,x2)
    g = calculateG(x1,x2)
    h = calculateH(x1,x2)

    return (a + b)/(g + h)
end

function S7(x1,x2)
    a = calculateA(x1,x2)
    b = calculateB(x1,x2)
    g = calculateG(x1,x2)
    h = calculateH(x1,x2)

    return (a * b - g * h)/(a * b + g * h)
end

function My(x1,x2)
    a = calculateA(x1,x2)
    b = calculateB(x1,x2)
    g = calculateG(x1,x2)
    h = calculateH(x1,x2)

    return (a + b)/(x1:size(1) + g + h)
end

function Hamming(x1,x2)
    result = 0
    for i = 1, x1:size(1) do
        if x1[i] ~= x2[i] then
            result = result + 1
        end
    end

    return result
end

watermelon = torch.Tensor({0,0,0,0,1,0,1})
cherryPlum = torch.Tensor({1,0,0,1,0,1,0})

print('Watermelon:')
print('cherry:')
print(S1(cherry,watermelon))
print('------------------')
print('orange:')
print(S1(orange,watermelon))
print('------------------')
print('apple:')
print(S1(apple,watermelon))
print('------------------')
print('melon:')
print(S1(melon,watermelon))
print('------------------')
print('banana:')
print(S1(banana,watermelon))
print('------------------')

print('####################')

print('Cherry-plum:')
print('cherry:')
print(S1(cherry,cherryPlum))
print('------------------')
print('orange:')
print(S1(orange,cherryPlum))
print('------------------')
print('apple:')
print(S1(apple,cherryPlum))
print('------------------')
print('melon:')
print(S1(melon,cherryPlum))
print('------------------')
print('banana:')
print(S1(banana,cherryPlum))
print('------------------')

print('###################')
print('Cherry-plum:')
print('cherry:')
print(My(cherry,cherryPlum))
print('------------------')
print('orange:')
print(My(orange,cherryPlum))
print('------------------')
print('apple:')
print(My(apple,cherryPlum))
print('------------------')
print('melon:')
print(My(melon,cherryPlum))
print('------------------')
print('banana:')
print(My(banana,cherryPlum))
print('------------------')

print('##################')
print('Maximum: apple with apple')
print(S1(apple,apple))
print('-----------------')
print('Minimum: cherry with banana')
print(S1(cherry,banana))

print('##################')
print('Hamming\'s distance between melon and cherry =',Hamming(melon,cherry)) 
print('Hamming\'s distance between orange and watermelon =',Hamming(orange,watermelon))
print('Hamming\'s distance between orange and banana =',Hamming(orange,banana))
notApple = torch.Tensor({0,1,0,1,0,1,1})
print('Maximum Hamming\'s distance =',Hamming(apple,notApple))

print('#################')
unripeMelon = torch.Tensor({0,0,0,0,1,0,1})
print('Cherry - Cherry-plum Hamming\'s distance =',Hamming(cherry,cherryPlum)) 
print('Cherry - Cherry-plum Kulzhinsky distance =',S6(cherry,cherryPlum))


import numpy as np
#np.random takes two parameters as parameters parameter1: row size, parameter2:column size
#the value will always between 0 and 1

wh=np.random.random_sample((4,3));
bh=np.random.random_sample((1,3));
wout=np.random.random_sample((3,1));
bout=np.random.random_sample((1,1));

print(wh);
print(bh);
print(wout);
print(bout);

x = [[1,0,1,0], [1,0,1,1], [0,1,0,1]];

wh = [[0.72810158, 0.81348652, 0.93678359],
 [0.69262494, 0.51311302, 0.78579618],
 [0.01547176, 0.12193186, 0.18553142],
 [0.63890484, 0.33646163, 0.19282042]];

bh = [[0.71505222, 0.45389456, 0.98751521],
    [0.71505222, 0.45389456, 0.98751521],
    [0.71505222, 0.45389456, 0.98751521]];

wout = [[0.73324102],
 [0.44269342],
 [0.32782907]];

bout = [[0.5478359], [0.5478359], [0.5478359]];

y = [[1],[1],[0]];

hidden_layer_input = np.dot(x,wh) + bh;


print("hidden_layer_input", hidden_layer_input);


def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x));

hiddenlayer_activations = sigmoid(hidden_layer_input);

output_layer_input = np.dot(hiddenlayer_activations, wout) + bout;

print("output_layer_input", output_layer_input);

output = sigmoid(output_layer_input);

print("output", output);

E = y - output;

print("E", E)

Slope_output_layer= sigmoid(output, True);

print("Slope_output_layer", Slope_output_layer);

Slope_hidden_layer = sigmoid(hiddenlayer_activations, True);

print("Slope_hidden_layer", Slope_hidden_layer);

lr = 0.1;

d_output = E * Slope_output_layer*lr;

print("d_output", d_output);

temp1 = np.array(wout);

Error_at_hidden_layer = np.dot(d_output, temp1.transpose());

print("Error_at_hidden_layer", Error_at_hidden_layer);

d_hiddenlayer = Error_at_hidden_layer * Slope_hidden_layer;

print("d_hiddenlayer", d_hiddenlayer);

temp2 = np.array(hiddenlayer_activations);

wout = wout + np.dot(temp2.transpose(), d_output) * lr;

print("wout", wout);

temp3 = np.array(x);

wh = wh + np.dot(temp3.transpose(),d_hiddenlayer) * lr;

print("wh", wh);

bh = bh + np.sum(d_hiddenlayer, axis=0) * lr;

print("bh", bh);

bout = bout + np.sum(d_output, axis=0)* lr;

print("bout", bout);


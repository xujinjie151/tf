import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import requests
HEADERS = {"User-Agent":r"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36"}
import re
from lxml import html
import grequests
import pandas as pd
pp1 = pd.DataFrame([], columns =  ["id", "time","value","hisvalue","dayrate"])

page_urls = []
for l in range(1,200):
    page_urls.append("http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=110011&page="+str(l))


rs = (grequests.get(url, headers = HEADERS) for url in page_urls)
responses = grequests.map(rs, size = 5)

for response in responses:
    fundRang = re.search("content:\"(.*?)\"", response.text, re.S).group(1)
    parsed_body = html.fromstring(fundRang)
    for k in range(1,11):
        fundName = parsed_body.xpath('//table//tbody//tr[' + str(k) + ']//text()')
        pp1.loc[len(pp1)] = {"id":len(pp1), "time":fundName[0] , "value":fundName[1] , "hisvalue":fundName[2] , "dayrate":fundName[3]}

pp1.to_csv("jijin.csv")

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
# pp = pd.read_csv("jijin.csv")
pp = pp1[:500]

pp["id"] = pp["id"].apply(lambda x :x/np.float32(1000.00))[::-1]

x_data = pp["id"].values[:, np.newaxis]

print(x_data[:10])
print(x_data.shape)
print("--------------")
# noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)

# print(noise.shape)
# print("--------------")

# y_data = np.square(x_data) - 0.5 + noise

y_data = pp["value"][::-1].values.astype(np.float32)[:, np.newaxis]

print(y_data[:10])

print(y_data.shape)
print("--------------")
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()  # 替换成这样就好

sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.ylim(1, 4)  
ax.scatter(x_data, y_data)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()

for i in range(30000):
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data, keep_prob: 1})
	if i % 50 == 0:
		print(sess.run(loss, feed_dict={xs: x_data, ys: y_data, keep_prob: 1}))
		# to visualize the result and improvement
		try:
			ax.lines.remove(lines[0])
		except Exception:
		    pass
		prediction_value = sess.run(prediction, feed_dict={xs: x_data, keep_prob: 1})
		# plot the prediction
		lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
		plt.pause(0.1)
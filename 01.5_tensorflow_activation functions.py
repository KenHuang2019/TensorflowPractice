
#activation functions用於日常中不能使用線性(linear)方程式解決的問題
#y=Wx  預測值=常數*輸入值
#y=AF(Wx) 預測值=激勵函數*(常數*輸入值)
#AF 可能產生梯度消失、爆炸的問題，若神經網絡的隱藏層較多層則須慎重考慮
#作用方式是將部分神經元激活後，將激活後的訊息傳遞至後方神經元
#可參考https://www.tensorflow.org/api_guides/python/nn使用，每種皆有其適用問題，需針對程序再進一步了解

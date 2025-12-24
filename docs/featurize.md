# featurize
云GPU平台 [featurize](https://featurize.cn/)

# 使用流程
## 进入主页

<img src="./imgs/Featurize/main.png" />

点击**开始使用**按钮

## 选择相应的GPU

<img src="./imgs/Featurize/GPUs.png" />

推荐使用RTX 4090，RTX 3090等应该也可，主要关注GPU显存大小，确保代码运行都跑得动、不会出现类似'CudaOutOfMemory'的报错就OK

<img src="./imgs/Featurize/selectGPU.png" />

点击**开始使用**按钮即可

## 进入工作区

<img src="./imgs/Featurize/waitForWorkspace.png" />

等待工作区初始化

<img src="./imgs/Featurize/enterWorkspace.png" />

初始化完成后，点击**打开工作区**按钮，进入工作区，这是一个类似vscode等IDE的界面

## 界面功能简介

<img src="./imgs/Featurize/workspace.png" />

蓝色加号那一排（左上角）对应功能分别为：新建文件、新建文件夹、上传文件、刷新

服务器每次**退还再租用**会"刷机"，仅**work文件夹**中的东西不会被清理，其他的东西都会被清除，所以所有操作都建议在work/文件夹下进行

// 这里我的work已经有东西了，为了方便演示，我就新建了一个work2/文件夹，正常的话就用work文件夹就可以了

服务器**强制重启**再打开应该没什么问题，不会"刷机"

## 退还服务器

<img src="./imgs/Featurize/enterWorkspace.png" />

每次不用了记得点击**退还实例**按钮，否则会一直挂着扣费！

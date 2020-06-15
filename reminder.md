## 关于数据

### 交通拥堵指数数据（train_TTI.csv）

-   id_road：枚举，路段编号
-   TTI：浮点，拥堵指数
-   speed：浮点，平均车速
-   time：时间，记录时间

### 网约车轨迹数据

-   id_order：字符串，订单编号
-   id_user：字符串，司机编号
-   gps_records：五个字段一组
    -   lng：经度
    -   lat：纬度
    -   speed：速度，单位为m/s
    -   direction：方向
    -   time：记录时间，时区为UTC

## 目前的一些参考

https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201701&filename=1016915036.nh&v=MTI5MTBIUHFaRWJQSVI4ZVgxTHV4WVM3RGgxVDNxVHJXTTFGckNVUjdxZlllWnNGQ2prVmI3T1ZGMjZHTHE1Rzk=

基于出租车GPS数据的城市交通拥堵识别和关联性分析

https://github.com/Michelia-zhx/Huaweicloud_Competition_Traffic/blob/master/clean_data

某个人的源代码

## 一些想法
首先，速度是最简单，也是最有效的特征。

关键在于地理位置，和街道的欧氏（曼哈顿？）距离，以及确定车辆的团簇

时间可能也是一个重要的指标，比如上下班时间可能堵车，，比如工作日和周末的拥堵程度不同。但是又要防止学习器只学习时间导致过拟合。

可以取距离街道在某一个曼哈顿距离的范数圆中的所有车辆，车辆总数，计算数量，平均速度等等。范数圆是否够好呢？街道是狭长的区域。（但也许并不需要那么复杂）

取所有车辆的时候要注意样本是一辆车的轨迹，所以计算车辆相关的属性时需要有点技巧。

在生成样本集的时候，采取怎样的时间截面是一个问题。截面太小容易漏掉一些车，界面太大同一辆车会多次出现。

要注意如果要将一些样本的曼哈顿距离加和的话，要考虑车辆的数量。但又要注意本来距离和大的话，就蕴含车辆数的信息。

另外一个问题是真的要考虑加和曼哈顿距离吗？有必要考虑车辆和街道的距离吗？或者说用于车辆的加权？要考虑一下现实的情况。

当前街道的tti也和附近的街道的tti相关。同样，当前街道和附近街道的速度特征也是都要考虑的。

关于时序，可以用RNN处理。
## 存在的疑问
听说出租车的速度和给出的路段速度无关？

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
    -   speed：速度
    -   direction：方向
    -   time：记录时间

## 目前的一些参考

https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201701&filename=1016915036.nh&v=MTI5MTBIUHFaRWJQSVI4ZVgxTHV4WVM3RGgxVDNxVHJXTTFGckNVUjdxZlllWnNGQ2prVmI3T1ZGMjZHTHE1Rzk=

基于出租车GPS数据的城市交通拥堵识别和关联性分析

https://github.com/Michelia-zhx/Huaweicloud_Competition_Traffic/blob/master/clean_data

某个人的源代码

## 一些想法
关键在于地理位置，和街道的欧氏（曼哈顿？）距离，以及确定车辆的团簇

时间可能也是一个重要的指标

可以取距离街道在某一个曼哈顿距离的范数圆中的所有车辆，计算数量，平均速度等等。

## 存在的疑问
utc时间？

# 智能图片内容识别服务

### 特性

- 后端：fastapi框架 pytorch环境 blip模型
- 主要功能模块：提供智能图片内容识别服务

### 安装

```
//镜像较大，加载可能比较耗时，请耐心等待
docker-compose up -d
```

### 接口

```POST http://ip:2233/api/predict```

字段设置

| 字段   | 是否必须 | 类型   | 描述   |
|------|------|------|------|
| file | 是    | file | 表单文件 |

请求返回：

```
{
    "code": 200,
    "message": "调用成功",
    "data": {
        "result": "a great white shark swims in the ocean"
    }
}
```

# 使用截图

- 以下图片识别为：`a great white shark swimming in the ocean`
  ![识别体验](test.jpg "使用截图")



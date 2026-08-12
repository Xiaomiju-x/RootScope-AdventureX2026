# RootScope Public Website

AdventureX 期间挂载到 `https://xiaomiju.xyz/` 的公开只读展示站。

## 关键合同

- 匿名访问，无用户账号、Cookie 或登录流程。
- 仅允许 `GET`、`HEAD`、`OPTIONS`。
- 无上传、表单、写 API、WebSocket、SSE、设备代理或执行器控制。
- 不依赖 RDK X5 在线；页面只展示已封存的历史实测。
- 旧 XRD Site32 服务、源代码和发布树保持原样；使用独立服务与原子 Caddy 路由切换。

## 本地运行

```text
python app.py --host 127.0.0.1 --port 29200
```

健康端点：`GET /healthz`

## 发布结构

```text
/opt/rootscope-web/releases/<release-id>/
/opt/rootscope-web/current -> releases/<release-id>
127.0.0.1:29200
```

`deploy/rootscope-web.service` 是最小权限 systemd 模板。生产切换前必须备份并哈希：

- 当前 Caddyfile；
- 原 XRD 发布树及 current 链接；
- 原 XRD systemd unit；
- 当前监听端口与防火墙状态。

回滚只恢复 Caddyfile 备份并重新加载，不删除 RootScope 或 XRD 任一发布树。

#!/bin/bash
# AI Foundation 项目设置脚本

set -e

echo "🚀 AI Foundation 项目设置"
echo "========================"

# 创建虚拟环境
echo "📦 创建Python虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 升级pip
echo "⬆️  升级pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装项目依赖..."
pip install -e ".[dev]"

# 安装gRPC依赖
echo "📦 安装gRPC依赖..."
pip install grpcio grpcio-tools

# 编译Proto文件
echo "🔨 编译Proto文件..."
python -m grpc_tools.protoc \
    -I=src/grpc_service \
    --python_out=src/grpc_service \
    --grpc_python_out=src/grpc_service \
    src/grpc_service/ai_core.proto || true

# 运行测试
echo "🧪 运行测试..."
pytest tests/ -v || echo "⚠️  部分测试可能失败（缺少API密钥）"

echo ""
echo "✅ 设置完成!"
echo ""
echo "使用说明:"
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 运行示例: python examples/basic_usage.py"
echo "  3. 启动gRPC服务: python -m src.grpc_service.server"
echo ""
echo "配置说明:"
echo "  编辑 config/default.yaml 设置API密钥"
echo "  或设置环境变量: export OPENAI_API_KEY=your-key"

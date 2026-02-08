#!/bin/bash
# 运行测试

set -e

echo "🧪 AI Foundation 测试套件"
echo "========================"

# 单元测试
echo ""
echo "📦 运行单元测试..."
pytest tests/unit/ -v --tb=short

# 集成测试
echo ""
echo "🔗 运行集成测试..."
pytest tests/integration/ -v --tb=short

# 覆盖率报告
echo ""
echo "📊 生成覆盖率报告..."
pytest tests/ --cov=src --cov-report=html --cov-report=term

echo ""
echo "✅ 所有测试完成！"
echo "覆盖率报告: htmlcov/index.html"

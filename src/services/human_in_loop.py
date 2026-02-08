# 人在回路服务
# Human-in-the-Loop 服务

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio


class HumanAction(Enum):
    """人工操作类型"""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    CONTINUE = "continue"
    STOP = "stop"


@dataclass
class HumanReview:
    """人工审查请求"""
    review_id: str
    agent_name: str
    task: str
    current_output: str
    context: Dict[str, Any]
    created_at: datetime
    status: str = "pending"  # pending, approved, rejected, modified
    human_action: Optional[HumanAction] = None
    human_comment: Optional[str] = None
    human_feedback: Optional[str] = None


class HumanInLoop:
    """人在回路服务"""
    
    def __init__(self):
        self._pending_reviews: Dict[str, HumanReview] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._enabled = True
    
    def enable(self):
        """启用人在回路"""
        self._enabled = True
    
    def disable(self):
        """禁用人在回路"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled
    
    async def request_review(
        self,
        agent_name: str,
        task: str,
        current_output: str,
        context: Optional[Dict] = None,
        callback: Optional[Callable] = None
    ) -> HumanReview:
        """请求人工审查"""
        import uuid
        
        review = HumanReview(
            review_id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            task=task,
            current_output=current_output,
            context=context or {},
            created_at=datetime.utcnow(),
        )
        
        self._pending_reviews[review.review_id] = review
        
        if callback:
            self._callbacks[review.review_id] = callback
        
        # 模拟等待人工响应
        # 在实际实现中，这里会暂停执行并等待人工输入
        if self._enabled:
            review = await self._wait_for_human(review)
        
        return review
    
    async def _wait_for_human(self, review: HumanReview) -> HumanReview:
        """等待人工响应"""
        # 在实际系统中，这里会发送通知并等待
        # 简化版本：使用超时或模拟
        
        # 模拟：如果5秒内没有人工响应，自动继续
        try:
            await asyncio.wait_for(
                self._check_for_response(review.review_id),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # 超时：自动批准继续
            review.status = "approved"
            review.human_action = HumanAction.CONTINUE
            review.human_comment = "Timeout - auto approved"
        
        return review
    
    async def _check_for_response(self, review_id: str):
        """检查是否有响应"""
        while review_id in self._pending_reviews:
            await asyncio.sleep(0.5)
    
    def submit_review(
        self,
        review_id: str,
        action: HumanAction,
        comment: Optional[str] = None,
        modified_output: Optional[str] = None
    ) -> bool:
        """提交审查结果"""
        if review_id not in self._pending_reviews:
            return False
        
        review = self._pending_reviews[review_id]
        review.status = action.value
        review.human_action = action
        review.human_comment = comment
        review.human_feedback = modified_output
        
        # 调用回调
        if review_id in self._callbacks:
            callback = self._callbacks[review_id]
            callback(review)
            del self._callbacks[review_id]
        
        return True
    
    def approve(self, review_id: str, comment: Optional[str] = None) -> bool:
        """批准"""
        return self.submit_review(review_id, HumanAction.APPROVE, comment)
    
    def reject(self, review_id: str, comment: Optional[str] = None) -> bool:
        """拒绝"""
        return self.submit_review(review_id, HumanAction.REJECT, comment)
    
    def modify(
        self,
        review_id: str,
        modified_output: str,
        comment: Optional[str] = None
    ) -> bool:
        """修改"""
        return self.submit_review(
            review_id,
            HumanAction.MODIFY,
            comment,
            modified_output
        )
    
    def get_pending_reviews(self) -> List[HumanReview]:
        """获取待处理审查"""
        return [
            review for review in self._pending_reviews.values()
            if review.status == "pending"
        ]
    
    def get_review(self, review_id: str) -> Optional[HumanReview]:
        """获取审查详情"""
        return self._pending_reviews.get(review_id)
    
    def cancel_review(self, review_id: str) -> bool:
        """取消审查"""
        if review_id in self._pending_reviews:
            del self._pending_reviews[review_id]
            if review_id in self._callbacks:
                del self._callbacks[review_id]
            return True
        return False


class ApprovalLevel(Enum):
    """审批级别"""
    NONE = 0
    LOW_RISK = 1
    MEDIUM_RISK = 2
    HIGH_RISK = 3


class ApprovalManager:
    """审批管理器 - 基于风险级别的审批流程"""
    
    def __init__(self, human_in_loop: Optional[HumanInLoop] = None):
        self.human_in_loop = human_in_loop or HumanInLoop()
        self._risk_rules: List[Callable] = []
    
    def add_risk_rule(self, rule: Callable[[Dict], ApprovalLevel]):
        """添加风险评估规则"""
        self._risk_rules.append(rule)
    
    async def check_approval(
        self,
        agent_name: str,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """检查是否需要审批"""
        # 评估风险级别
        max_level = ApprovalLevel.NONE
        for rule in self._risk_rules:
            level = rule(context)
            if level.value > max_level.value:
                max_level = level
        
        # 根据风险级别决定
        if max_level == ApprovalLevel.NONE:
            return {"approved": True, "level": "none", "reason": "Low risk"}
        
        if max_level == ApprovalLevel.HIGH_RISK:
            # 高风险：必须人工审批
            review = await self.human_in_loop.request_review(
                agent_name=agent_name,
                task=task,
                current_output=context.get("current_output", ""),
                context=context,
            )
            
            return {
                "approved": review.status == "approved",
                "review_id": review.review_id,
                "level": "high",
                "action": review.human_action.value if review.human_action else None,
            }
        
        # 中等风险：可以选择审批或自动继续
        return {
            "approved": True,  # 自动批准
            "level": "medium",
            "warning": "Medium risk action - consider review",
        }

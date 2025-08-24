"""
Dataset for benchmarking the CoachUp AI system.

Contains realistic conversation scenarios covering:
- Single-turn interactions (one_off assessments)
- Multi-turn interactions (start/continue/end patterns)
- Different skill levels and assessment criteria
- Edge cases and challenging scenarios
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class SkillLevel(Enum):
    """Skill levels from PRD2 for Clarity/Eloquence"""
    NOVICE_1_3 = "1-3"  # Often confusing or disorganized
    DEVELOPING_4_6 = "4-6"  # Generally understandable
    PROFICIENT_7_8 = "7-8"  # Clear, direct; simplifies complexity
    MASTER_9_10 = "9-10"  # Effortless, memorable communication


class BoundaryDecision(Enum):
    """Expected classifier decisions"""
    START = "start"
    CONTINUE = "continue"
    END = "end"
    ONE_OFF = "one_off"
    IGNORE = "ignore"
    ABSTAIN = "abstain"


@dataclass
class ConversationMessage:
    """A single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    message_id: str
    timestamp: int


@dataclass
class ConversationScenario:
    """A complete conversation scenario with expected outcomes"""
    scenario_id: str
    description: str
    skill_focus: str  # e.g., "clarity", "persuasiveness", "grammar"
    expected_skill_level: SkillLevel
    messages: List[ConversationMessage]
    expected_boundary_decisions: List[Tuple[str, BoundaryDecision, float]]  # (message_id, decision, confidence)
    expected_assessment_scores: Dict[str, float]  # rubric category -> score (0-1)
    expected_focus_areas: List[Dict[str, Any]]  # [{"title": str, "action": str, "priority": str}]
    tags: List[str]  # For filtering scenarios


class BenchmarkDataset:
    """Collection of conversation scenarios for benchmarking"""

    def __init__(self):
        self.scenarios = self._create_scenarios()

    def _create_scenarios(self) -> List[ConversationScenario]:
        """Create all benchmark scenarios"""
        return [
            # Single-turn scenarios (one_off)
            self._create_single_turn_clarity_novice(),
            self._create_single_turn_clarity_master(),
            self._create_single_turn_persuasion_weak(),
            self._create_single_turn_persuasion_strong(),

            # Multi-turn scenarios
            self._create_multi_turn_sales_pitch_novice(),
            self._create_multi_turn_sales_pitch_proficient(),
            self._create_multi_turn_interview_preparation(),
            self._create_multi_turn_technical_explanation(),

            # Edge cases
            self._create_edge_case_very_short(),
            self._create_edge_case_mixed_quality(),
            self._create_edge_case_technical_jargon(),
            self._create_edge_case_emotional_content(),
        ]

    def _create_single_turn_clarity_novice(self) -> ConversationScenario:
        """Single turn with novice-level clarity issues (from PRD2 example)"""
        return ConversationScenario(
            scenario_id="single_clarity_novice",
            description="Single turn with confusing jargon and poor organization",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.NOVICE_1_3,
            messages=[
                ConversationMessage(
                    role="user",
                    content="So, we’ve implemented a new synergistic paradigm leveraging our backend architecture to optimize the user experience metrics.",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.85)
            ],
            expected_assessment_scores={
                "correctness": 0.75,  # Technically correct but confusing
                "clarity": 0.25,      # Very unclear due to jargon
                "conciseness": 0.35,  # Too wordy
                "fluency": 0.60       # Grammatically correct but awkward
            },
            expected_focus_areas=[
                {
                    "title": "Replace jargon with simple terms",
                    "action": "Use 'we made a change to how the app gets data' instead of technical terms",
                    "priority": "high"
                },
                {
                    "title": "Be more concise",
                    "action": "Remove unnecessary words and get to the point faster",
                    "priority": "medium"
                }
            ],
            tags=["single_turn", "clarity", "novice", "jargon"]
        )

    def _create_single_turn_clarity_master(self) -> ConversationScenario:
        """Single turn with master-level clarity (from PRD2 example)"""
        return ConversationScenario(
            scenario_id="single_clarity_master",
            description="Single turn with effortless, memorable communication",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.MASTER_9_10,
            messages=[
                ConversationMessage(
                    role="user",
                    content="The app is instantly responsive because data loads silently in the background.",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.82)
            ],
            expected_assessment_scores={
                "correctness": 0.95,
                "clarity": 0.98,
                "conciseness": 0.90,
                "fluency": 0.95
            },
            expected_focus_areas=[],  # Master level - no improvement areas needed
            tags=["single_turn", "clarity", "master", "exemplary"]
        )

    def _create_single_turn_persuasion_weak(self) -> ConversationScenario:
        """Single turn with weak persuasive language"""
        return ConversationScenario(
            scenario_id="single_persuasion_weak",
            description="Weak sales pitch with hedging and lack of confidence",
            skill_focus="persuasiveness",
            expected_skill_level=SkillLevel.NOVICE_1_3,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Um, so I guess you might want to buy this water bottle? It could be useful sometimes?",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.78)
            ],
            expected_assessment_scores={
                "correctness": 0.70,
                "clarity": 0.65,
                "conciseness": 0.55,
                "fluency": 0.45  # Lots of hedging and uncertainty
            },
            expected_focus_areas=[
                {
                    "title": "Remove hedging language",
                    "action": "Replace 'um', 'I guess', 'might', 'could' with direct, confident statements",
                    "priority": "high"
                },
                {
                    "title": "Focus on benefits",
                    "action": "Emphasize what the product does for the customer, not just features",
                    "priority": "high"
                }
            ],
            tags=["single_turn", "persuasiveness", "novice", "hedging"]
        )

    def _create_single_turn_persuasion_strong(self) -> ConversationScenario:
        """Single turn with strong persuasive language"""
        return ConversationScenario(
            scenario_id="single_persuasion_strong",
            description="Strong, confident sales pitch",
            skill_focus="persuasiveness",
            expected_skill_level=SkillLevel.MASTER_9_10,
            messages=[
                ConversationMessage(
                    role="user",
                    content="This water bottle will keep your drinks perfectly cold for 24 hours, so you can stay focused and hydrated during your busiest days.",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.81)
            ],
            expected_assessment_scores={
                "correctness": 0.92,
                "clarity": 0.94,
                "conciseness": 0.88,
                "fluency": 0.91
            },
            expected_focus_areas=[],  # Strong performance
            tags=["single_turn", "persuasiveness", "master", "confident"]
        )

    def _create_multi_turn_sales_pitch_novice(self) -> ConversationScenario:
        """Multi-turn sales conversation with novice-level performance"""
        return ConversationScenario(
            scenario_id="multi_sales_novice",
            description="Novice-level multi-turn sales conversation with frequent hedging",
            skill_focus="persuasiveness",
            expected_skill_level=SkillLevel.NOVICE_1_3,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Uh... I don't know if you're interested, but maybe you'd like to hear about this water bottle?",
                    message_id="msg_001",
                    timestamp=1692540000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="I'm listening. What makes this water bottle special?",
                    message_id="msg_002",
                    timestamp=1692540001000
                ),
                ConversationMessage(
                    role="user",
                    content="Well, um, it keeps drinks cold for like 12 hours or something. I think that's pretty good?",
                    message_id="msg_003",
                    timestamp=1692540002000
                ),
                ConversationMessage(
                    role="assistant",
                    content="That's interesting. How does that benefit me specifically?",
                    message_id="msg_004",
                    timestamp=1692540003000
                ),
                ConversationMessage(
                    role="user",
                    content="You could use it for your daily commute or when you work out. I suppose it might be convenient?",
                    message_id="msg_005",
                    timestamp=1692540004000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.START, 0.73),
                ("msg_002", BoundaryDecision.CONTINUE, 0.88),
                ("msg_003", BoundaryDecision.CONTINUE, 0.85),
                ("msg_004", BoundaryDecision.CONTINUE, 0.82),
                ("msg_005", BoundaryDecision.END, 0.76)
            ],
            expected_assessment_scores={
                "correctness": 0.65,
                "clarity": 0.55,
                "conciseness": 0.60,
                "fluency": 0.35  # Heavy use of hedging throughout
            },
            expected_focus_areas=[
                {
                    "title": "Eliminate hedging completely",
                    "action": "Remove all instances of 'uh', 'um', 'maybe', 'I think', 'I suppose', 'or something'",
                    "priority": "high"
                },
                {
                    "title": "Focus on specific benefits",
                    "action": "Instead of 'it might be convenient', say 'it eliminates the need to buy drinks during your commute'",
                    "priority": "high"
                }
            ],
            tags=["multi_turn", "persuasiveness", "novice", "hedging", "sales"]
        )

    def _create_multi_turn_sales_pitch_proficient(self) -> ConversationScenario:
        """Multi-turn sales conversation with proficient-level performance"""
        return ConversationScenario(
            scenario_id="multi_sales_proficient",
            description="Proficient-level multi-turn sales conversation",
            skill_focus="persuasiveness",
            expected_skill_level=SkillLevel.PROFICIENT_7_8,
            messages=[
                ConversationMessage(
                    role="user",
                    content="This water bottle keeps drinks cold for 24 hours and has a comfortable grip for daily use.",
                    message_id="msg_001",
                    timestamp=1692540000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="That sounds practical. What makes it better than other bottles on the market?",
                    message_id="msg_002",
                    timestamp=1692540001000
                ),
                ConversationMessage(
                    role="user",
                    content="Unlike most bottles, this one won't sweat on your desk or in your bag, and it's designed specifically for people who are always on the move.",
                    message_id="msg_003",
                    timestamp=1692540002000
                ),
                ConversationMessage(
                    role="assistant",
                    content="I like that it doesn't sweat. How much does it cost?",
                    message_id="msg_004",
                    timestamp=1692540003000
                ),
                ConversationMessage(
                    role="user",
                    content="It's $35, and that includes a 5-year warranty. The peace of mind alone is worth the investment.",
                    message_id="msg_005",
                    timestamp=1692540004000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.START, 0.71),
                ("msg_002", BoundaryDecision.CONTINUE, 0.86),
                ("msg_003", BoundaryDecision.CONTINUE, 0.83),
                ("msg_004", BoundaryDecision.CONTINUE, 0.79),
                ("msg_005", BoundaryDecision.END, 0.74)
            ],
            expected_assessment_scores={
                "correctness": 0.85,
                "clarity": 0.82,
                "conciseness": 0.78,
                "fluency": 0.80
            },
            expected_focus_areas=[
                {
                    "title": "Add more specific examples",
                    "action": "Include concrete scenarios where the product makes a real difference",
                    "priority": "medium"
                }
            ],
            tags=["multi_turn", "persuasiveness", "proficient", "sales"]
        )

    def _create_multi_turn_interview_preparation(self) -> ConversationScenario:
        """Multi-turn interview preparation scenario"""
        return ConversationScenario(
            scenario_id="multi_interview_prep",
            description="Multi-turn interview preparation with mixed performance",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.DEVELOPING_4_6,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Can you help me prepare for a job interview? I need to talk about my experience.",
                    message_id="msg_001",
                    timestamp=1692540000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="Of course! Tell me about your most recent role and what you accomplished there.",
                    message_id="msg_002",
                    timestamp=1692540001000
                ),
                ConversationMessage(
                    role="user",
                    content="So, I was working at this company, and I did a lot of stuff. Like, I managed projects and talked to clients. It was pretty good.",
                    message_id="msg_003",
                    timestamp=1692540002000
                ),
                ConversationMessage(
                    role="assistant",
                    content="That sounds interesting. Can you give me a specific example of a project you managed?",
                    message_id="msg_004",
                    timestamp=1692540003000
                ),
                ConversationMessage(
                    role="user",
                    content="There was this one project where I had to coordinate with the team. We built a new feature for the app. It took about three months and everyone was happy with the result.",
                    message_id="msg_005",
                    timestamp=1692540004000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.START, 0.69),
                ("msg_002", BoundaryDecision.CONTINUE, 0.81),
                ("msg_003", BoundaryDecision.CONTINUE, 0.77),
                ("msg_004", BoundaryDecision.CONTINUE, 0.75),
                ("msg_005", BoundaryDecision.END, 0.72)
            ],
            expected_assessment_scores={
                "correctness": 0.75,
                "clarity": 0.65,  # Vague descriptions like "did a lot of stuff"
                "conciseness": 0.70,
                "fluency": 0.68
            },
            expected_focus_areas=[
                {
                    "title": "Use specific examples",
                    "action": "Instead of 'did a lot of stuff', describe concrete achievements with numbers and outcomes",
                    "priority": "high"
                },
                {
                    "title": "Show impact",
                    "action": "Explain how your work benefited the company or team, not just what you did",
                    "priority": "medium"
                }
            ],
            tags=["multi_turn", "clarity", "interview", "developing", "vague"]
        )

    def _create_multi_turn_technical_explanation(self) -> ConversationScenario:
        """Multi-turn technical explanation scenario"""
        return ConversationScenario(
            scenario_id="multi_technical_explanation",
            description="Technical explanation that needs simplification",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.PROFICIENT_7_8,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Let me explain how our new authentication system works.",
                    message_id="msg_001",
                    timestamp=1692540000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="I'm listening. How does it improve security?",
                    message_id="msg_002",
                    timestamp=1692540001000
                ),
                ConversationMessage(
                    role="user",
                    content="We implemented OAuth 2.0 with PKCE extension to prevent authorization code interception attacks. The system now uses JWT tokens with RS256 signatures instead of the previous HS256 implementation.",
                    message_id="msg_003",
                    timestamp=1692540002000
                ),
                ConversationMessage(
                    role="assistant",
                    content="That sounds technical. How does this affect our users?",
                    message_id="msg_004",
                    timestamp=1692540003000
                ),
                ConversationMessage(
                    role="user",
                    content="Users will now have a more secure login experience. Instead of remembering passwords, they can use their Google or Apple accounts, and the system is much less vulnerable to hackers stealing their session information.",
                    message_id="msg_005",
                    timestamp=1692540004000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.START, 0.67),
                ("msg_002", BoundaryDecision.CONTINUE, 0.79),
                ("msg_003", BoundaryDecision.CONTINUE, 0.76),
                ("msg_004", BoundaryDecision.CONTINUE, 0.73),
                ("msg_005", BoundaryDecision.END, 0.70)
            ],
            expected_assessment_scores={
                "correctness": 0.88,
                "clarity": 0.75,  # Good at adapting from technical to user-friendly
                "conciseness": 0.72,
                "fluency": 0.78
            },
            expected_focus_areas=[
                {
                    "title": "Adapt technical depth to audience",
                    "action": "Start with user benefits before diving into technical details",
                    "priority": "medium"
                }
            ],
            tags=["multi_turn", "clarity", "technical", "proficient", "adaptation"]
        )

    def _create_edge_case_very_short(self) -> ConversationScenario:
        """Edge case: Very short, unclear message"""
        return ConversationScenario(
            scenario_id="edge_very_short",
            description="Very short message that should trigger one_off assessment",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.NOVICE_1_3,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Yeah",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.91)
            ],
            expected_assessment_scores={
                "correctness": 0.60,  # Minimal content to assess
                "clarity": 0.20,      # Extremely unclear what is being communicated
                "conciseness": 0.95,  # Very concise but lacks substance
                "fluency": 0.70
            },
            expected_focus_areas=[
                {
                    "title": "Provide complete answers",
                    "action": "Expand on your thoughts with specific details and context",
                    "priority": "high"
                }
            ],
            tags=["edge_case", "single_turn", "very_short", "unclear"]
        )

    def _create_edge_case_mixed_quality(self) -> ConversationScenario:
        """Edge case: Mixed quality throughout conversation"""
        return ConversationScenario(
            scenario_id="edge_mixed_quality",
            description="Conversation with varying quality across turns",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.DEVELOPING_4_6,
            messages=[
                ConversationMessage(
                    role="user",
                    content="The new dashboard provides real-time analytics that help users make informed decisions quickly.",
                    message_id="msg_001",
                    timestamp=1692540000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="That sounds great! Can you show me how it works?",
                    message_id="msg_002",
                    timestamp=1692540001000
                ),
                ConversationMessage(
                    role="user",
                    content="Uh, basically it's like, you know, the graphs and stuff show data and users can click around and see things.",
                    message_id="msg_003",
                    timestamp=1692540002000
                ),
                ConversationMessage(
                    role="assistant",
                    content="I need more specific details to understand the full value proposition.",
                    message_id="msg_004",
                    timestamp=1692540003000
                ),
                ConversationMessage(
                    role="user",
                    content="The dashboard displays key performance indicators with interactive charts, allowing stakeholders to drill down into specific metrics and time periods for comprehensive analysis.",
                    message_id="msg_005",
                    timestamp=1692540004000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.START, 0.65),
                ("msg_002", BoundaryDecision.CONTINUE, 0.78),
                ("msg_003", BoundaryDecision.CONTINUE, 0.75),
                ("msg_004", BoundaryDecision.CONTINUE, 0.72),
                ("msg_005", BoundaryDecision.END, 0.68)
            ],
            expected_assessment_scores={
                "correctness": 0.80,
                "clarity": 0.65,  # Inconsistent quality - great start and end, poor middle
                "conciseness": 0.75,
                "fluency": 0.60   # Mixed between professional and casual language
            },
            expected_focus_areas=[
                {
                    "title": "Maintain consistent quality",
                    "action": "Avoid filler words like 'uh', 'basically', 'like', 'you know', 'and stuff' throughout the entire conversation",
                    "priority": "high"
                }
            ],
            tags=["edge_case", "multi_turn", "mixed_quality", "inconsistent"]
        )

    def _create_edge_case_technical_jargon(self) -> ConversationScenario:
        """Edge case: Heavy technical jargon that needs simplification"""
        return ConversationScenario(
            scenario_id="edge_technical_jargon",
            description="Technical explanation that alienates non-technical audience",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.NOVICE_1_3,
            messages=[
                ConversationMessage(
                    role="user",
                    content="Our microservices architecture leverages container orchestration via Kubernetes to ensure horizontal scalability and fault tolerance across the distributed system topology.",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.84)
            ],
            expected_assessment_scores={
                "correctness": 0.85,  # Technically accurate
                "clarity": 0.15,      # Extremely unclear to general audience
                "conciseness": 0.40,  # Uses many words to say something simple
                "fluency": 0.75
            },
            expected_focus_areas=[
                {
                    "title": "Use everyday language",
                    "action": "Replace 'microservices architecture leverages container orchestration via Kubernetes' with 'we built the system to run smoothly and handle more users automatically'",
                    "priority": "high"
                }
            ],
            tags=["edge_case", "single_turn", "jargon", "technical"]
        )

    def _create_edge_case_emotional_content(self) -> ConversationScenario:
        """Edge case: Emotional content affecting communication quality"""
        return ConversationScenario(
            scenario_id="edge_emotional",
            description="Emotional conversation that impacts clarity",
            skill_focus="clarity",
            expected_skill_level=SkillLevel.DEVELOPING_4_6,
            messages=[
                ConversationMessage(
                    role="user",
                    content="I absolutely love this new feature! It's amazing and works perfectly every time without any issues whatsoever!",
                    message_id="msg_001",
                    timestamp=1692540000000
                )
            ],
            expected_boundary_decisions=[
                ("msg_001", BoundaryDecision.ONE_OFF, 0.79)
            ],
            expected_assessment_scores={
                "correctness": 0.70,
                "clarity": 0.55,  # Over-enthusiasm reduces clarity
                "conciseness": 0.25,  # Very wordy and repetitive
                "fluency": 0.65
            },
            expected_focus_areas=[
                {
                    "title": "Balance enthusiasm with clarity",
                    "action": "Reduce excessive adjectives and repetition while maintaining positive tone",
                    "priority": "medium"
                }
            ],
            tags=["edge_case", "single_turn", "emotional", "over_enthusiastic"]
        )

    def get_scenarios_by_tags(self, tags: List[str]) -> List[ConversationScenario]:
        """Get scenarios that match any of the provided tags"""
        return [s for s in self.scenarios if any(tag in s.tags for tag in tags)]

    def get_scenarios_by_skill_level(self, level: SkillLevel) -> List[ConversationScenario]:
        """Get scenarios for a specific skill level"""
        return [s for s in self.scenarios if s.expected_skill_level == level]

    def get_single_turn_scenarios(self) -> List[ConversationScenario]:
        """Get all single-turn scenarios"""
        return [s for s in self.scenarios if "single_turn" in s.tags]

    def get_multi_turn_scenarios(self) -> List[ConversationScenario]:
        """Get all multi-turn scenarios"""
        return [s for s in self.scenarios if "multi_turn" in s.tags]

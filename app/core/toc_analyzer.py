"""
Theory of Constraints (TOC) Analyzer
=====================================
Identifies process bottlenecks and generates a 5-step TOC action plan
by analyzing step-level automation scores, pain points, and duration estimates.

TOC Five Focusing Steps:
  1. IDENTIFY   – Find the constraint (the bottleneck step)
  2. EXPLOIT     – Squeeze maximum throughput from it without big changes (quick wins)
  3. SUBORDINATE – Align all other steps to support the constraint
  4. ELEVATE     – Invest to break the constraint (high-effort, high-ROI)
  5. REPEAT      – Find the next constraint once this one is resolved
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConstraintStep:
    """A step identified as a bottleneck / constraint."""
    step_number: int
    title: str
    actor: str
    automation_potential: int          # 0-100 (lower = harder to automate = more constraint)
    pain_points: List[str]
    duration_estimate: Optional[str]
    step_type: str
    bottleneck_score: float            # composite score (higher = more constrained)
    bottleneck_reasons: List[str]      # human-readable reasons why this is a constraint


@dataclass
class TOCAction:
    """One concrete action item inside a TOC phase."""
    action: str                        # what to do
    owner: str                         # who should do it
    impact: str                        # high | medium | low
    effort: str                        # low | medium | high
    automation_type: Optional[str]     # rpa | ai_agent | workflow | none


@dataclass
class TOCPhaseResult:
    """Result for one of the 5 TOC focusing steps."""
    phase: int                         # 1-5
    name: str                          # IDENTIFY | EXPLOIT | SUBORDINATE | ELEVATE | REPEAT
    summary: str
    actions: List[TOCAction]
    target_steps: List[int]            # step numbers this phase focuses on


@dataclass
class TOCAnalysisResult:
    """Full TOC analysis output attached to a process."""
    process_key: str
    process_title: str
    primary_constraint: Optional[ConstraintStep]
    secondary_constraints: List[ConstraintStep]
    throughput_score: float            # 0-100: estimated process throughput (higher = better)
    bottleneck_index: float            # 0-100: how severe the constraint is (higher = worse)
    phases: List[TOCPhaseResult]
    summary: str
    improvement_potential_pct: float   # estimated % improvement if constraints are resolved


# ─────────────────────────────────────────────────────────────────────────────
# Core analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TOCAnalyzer:
    """
    Performs Theory of Constraints analysis on extracted process steps.

    Usage (inside AnalysisService.analyze):
        toc = TOCAnalyzer()
        toc_result = toc.analyze(
            process_key=process_doc._key,
            process_title=process_title,
            steps=raw_steps,            # list of dicts from LLM extraction
            suggestions=raw_suggestions # list of dicts from LLM suggestions
        )
    """

    # Weights for bottleneck scoring
    _W_LOW_AUTOMATION  = 0.40   # low automation = high constraint weight
    _W_PAIN_POINTS     = 0.30   # more pain points = higher constraint
    _W_MANUAL_TYPE     = 0.15   # manual/approval steps are inherently constrained
    _W_DURATION        = 0.15   # longer steps slow the whole chain

    _MANUAL_TYPES = {"manual", "approval", "Higher Human intervention"}

    def analyze(
        self,
        process_key: str,
        process_title: str,
        steps: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]],
    ) -> TOCAnalysisResult:
        """
        Run the full 5-step TOC focusing cycle.
        Returns a TOCAnalysisResult dataclass.
        """
        if not steps:
            logger.warning("TOC: no steps provided, returning empty result")
            return self._empty_result(process_key, process_title)

        # ── Phase 1: IDENTIFY ─────────────────────────────────────────────────
        ranked = self._rank_constraints(steps)
        primary   = ranked[0]  if ranked      else None
        secondary = ranked[1:3]  # top 2 additional constraints

        # ── Throughput & bottleneck metrics ──────────────────────────────────
        avg_automation = (
            sum(s.get("automation_potential", 50) for s in steps) / len(steps)
        )
        bottleneck_index = round(primary.bottleneck_score * 100, 1) if primary else 0.0
        throughput_score = round(avg_automation, 1)

        # Estimated improvement: resolving top constraint frees up that bottleneck's
        # share of the flow — modelled as (bottleneck_score / n_steps) * 100 + base lift
        n = len(steps)
        improvement_potential = round(
            min(95.0, (bottleneck_index / n) * 1.5 + 10), 1
        )

        # ── Build the 5 TOC phases ────────────────────────────────────────────
        phases = [
            self._phase_identify(primary, secondary),
            self._phase_exploit(primary, suggestions),
            self._phase_subordinate(primary, steps, suggestions),
            self._phase_elevate(primary, secondary, suggestions),
            self._phase_repeat(secondary),
        ]

        summary = self._build_summary(
            process_title, primary, throughput_score, improvement_potential
        )

        return TOCAnalysisResult(
            process_key=process_key,
            process_title=process_title,
            primary_constraint=primary,
            secondary_constraints=secondary,
            throughput_score=throughput_score,
            bottleneck_index=bottleneck_index,
            phases=phases,
            summary=summary,
            improvement_potential_pct=improvement_potential,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Constraint ranking
    # ─────────────────────────────────────────────────────────────────────────

    def _rank_constraints(self, steps: List[Dict]) -> List[ConstraintStep]:
        scored = []
        max_pain = max((len(s.get("pain_points", [])) for s in steps), default=1) or 1
        max_dur  = self._max_duration_hours(steps)

        for s in steps:
            score, reasons = self._bottleneck_score(s, max_pain, max_dur)
            scored.append(
                ConstraintStep(
                    step_number=int(s.get("step_number", 0)),
                    title=s.get("title", ""),
                    actor=s.get("actor", "Unknown"),
                    automation_potential=int(s.get("automation_potential", 50)),
                    pain_points=s.get("pain_points", []),
                    duration_estimate=s.get("duration_estimate"),
                    step_type=s.get("step_type", "manual"),
                    bottleneck_score=score,
                    bottleneck_reasons=reasons,
                )
            )

        return sorted(scored, key=lambda x: x.bottleneck_score, reverse=True)

    def _bottleneck_score(
        self,
        step: Dict,
        max_pain: int,
        max_dur_hours: float,
    ):
        """Composite bottleneck score in [0, 1]. Higher = more constrained."""
        reasons = []

        # 1. Low automation potential  (inverted: 0 auto = 1.0 contribution)
        auto = step.get("automation_potential", 50) / 100.0
        auto_contrib = (1.0 - auto) * self._W_LOW_AUTOMATION
        if auto < 0.5:
            reasons.append(
                f"Low automation potential ({step.get('automation_potential', 50)}%) — "
                "requires heavy human involvement"
            )

        # 2. Pain points density
        pain_count = len(step.get("pain_points", []))
        pain_contrib = (pain_count / max_pain) * self._W_PAIN_POINTS
        if pain_count > 0:
            reasons.append(
                f"{pain_count} identified pain point(s): "
                + "; ".join(step.get("pain_points", [])[:2])
            )

        # 3. Step type penalty
        step_type = step.get("step_type", "manual")
        is_manual = step_type in self._MANUAL_TYPES
        type_contrib = self._W_MANUAL_TYPE if is_manual else 0.0
        if is_manual:
            reasons.append(f"Step type '{step_type}' requires manual/human action")

        # 4. Duration penalty
        dur_hours = self._parse_duration_hours(step.get("duration_estimate"))
        if max_dur_hours > 0:
            dur_contrib = (dur_hours / max_dur_hours) * self._W_DURATION
        else:
            dur_contrib = 0.0
        if dur_hours > 4:
            reasons.append(f"Long duration estimate ({step.get('duration_estimate')})")

        

        
        # 5. 🔥 NEW: Micro-step bottleneck detection
        micro_contrib = 0.0
        micro_reasons = []

        micro_steps = step.get("micro_steps", [])

        if micro_steps:
            micro_scores = []

            for ms in micro_steps:
                ms_score = 0

                text = str(ms).lower()

                if "manual" in text:
                    ms_score += 0.2
                if "approval" in text:
                    ms_score += 0.25
                if "wait" in text or "delay" in text:
                    ms_score += 0.3

                micro_scores.append(ms_score)

            if micro_scores:
                worst = max(micro_scores)   # ✅ TOC principle
                micro_contrib = worst * 0.25   # weight

                if worst > 0.2:
                    micro_reasons.append("Critical micro-step bottleneck inside step")


            # 6. 🔥 NEW: Decision-based constraint
        decision_contrib = 0.0

        decisions = step.get("decisions", [])

        for d in decisions:
                dtype = d.get("decision_type")

                if dtype == "human_judgment":
                    decision_contrib += 0.25
                    reasons.append("Human judgment decision slows flow")

                elif dtype == "ai_possible":
                    decision_contrib += 0.1

        total = (
            auto_contrib
            + pain_contrib
            + type_contrib
            + dur_contrib
            + micro_contrib
            + decision_contrib
        )
        if not reasons:
            reasons.append("Moderate constraint based on combined factors")

        return round(total, 4), reasons

    # ─────────────────────────────────────────────────────────────────────────
    # TOC Phase builders
    # ─────────────────────────────────────────────────────────────────────────

    def _phase_identify(
        self,
        primary: Optional[ConstraintStep],
        secondary: List[ConstraintStep],
    ) -> TOCPhaseResult:
        if not primary:
            return TOCPhaseResult(
                phase=1, name="IDENTIFY",
                summary="No significant constraint detected.",
                actions=[], target_steps=[],
            )

        actions = [
            TOCAction(
                action=(
                    f"Flag Step {primary.step_number} — '{primary.title}' — as the primary "
                    f"constraint. Document all pain points: "
                    + ("; ".join(primary.pain_points[:3]) or "N/A")
                ),
                owner=primary.actor,
                impact="high", effort="low", automation_type=None,
            ),
            TOCAction(
                action=(
                    "Map the exact handoff points before and after this step to measure "
                    "wait time and queue depth."
                ),
                owner="Process Owner",
                impact="high", effort="low", automation_type=None,
            ),
        ]
        if secondary:
            actions.append(TOCAction(
                action=(
                    "Note secondary constraints: "
                    + ", ".join(f"Step {s.step_number} ({s.title})" for s in secondary)
                    + ". Address these after the primary is resolved."
                ),
                owner="Process Owner",
                impact="medium", effort="low", automation_type=None,
            ))

        return TOCPhaseResult(
            phase=1, name="IDENTIFY",
            summary=(
                f"Primary constraint is Step {primary.step_number} — '{primary.title}' "
                f"(actor: {primary.actor}). Bottleneck score: "
                f"{round(primary.bottleneck_score * 100, 1)}%. "
                "Reasons: " + "; ".join(primary.bottleneck_reasons[:2]) + "."
            ),
            actions=actions,
            target_steps=[primary.step_number] + [s.step_number for s in secondary],
        )

    def _phase_exploit(
        self,
        primary: Optional[ConstraintStep],
        suggestions: List[Dict],
    ) -> TOCPhaseResult:
        """
        EXPLOIT: squeeze more throughput from the constraint without big investments.
        Map quick-win automation suggestions to the primary bottleneck step.
        """
        if not primary:
            return TOCPhaseResult(
                phase=2, name="EXPLOIT",
                summary="No constraint identified — nothing to exploit.",
                actions=[], target_steps=[],
            )

        # Find suggestions that target the constraint step
        related_sug = [
            s for s in suggestions
            if int(s.get("step_number", -1)) == primary.step_number
        ]

        actions = []

        # Quick-win suggestions
        for sug in related_sug[:2]:
            actions.append(TOCAction(
                action=(
                    f"Implement: {sug.get('title', 'Automation suggestion')}. "
                    f"{sug.get('description', '')}"
                ),
                owner=primary.actor,
                impact=sug.get("roi_impact", "medium"),
                effort=sug.get("effort_level", "medium"),
                automation_type=sug.get("automation_type", "workflow"),
            ))

        # Generic exploit actions if no suggestions found
        if not actions:
            actions.append(TOCAction(
                action=(
                    f"Analyse the exact sub-tasks inside Step {primary.step_number} "
                    "and eliminate or simplify any non-value-adding micro-steps immediately."
                ),
                owner=primary.actor,
                impact="medium", effort="low", automation_type=None,
            ))

        actions.append(TOCAction(
            action=(
                "Reduce idle/wait time at this step by pre-staging all required inputs "
                "before the step begins (input pre-loading)."
            ),
            owner="Process Owner",
            impact="medium", effort="low", automation_type="workflow",
        ))

        return TOCPhaseResult(
            phase=2, name="EXPLOIT",
            summary=(
                f"Maximize throughput at Step {primary.step_number} — '{primary.title}' "
                "using quick-win changes that require minimal investment. "
                f"{len(related_sug)} automation suggestion(s) directly target this step."
            ),
            actions=actions,
            target_steps=[primary.step_number],
        )

    def _phase_subordinate(
        self,
        primary: Optional[ConstraintStep],
        steps: List[Dict],
        suggestions: List[Dict],
    ) -> TOCPhaseResult:
        """
        SUBORDINATE: tune all non-constraint steps to feed the constraint optimally.
        Prevent upstream steps from over-producing and downstream steps from starving.
        """
        if not primary:
            return TOCPhaseResult(
                phase=3, name="SUBORDINATE",
                summary="No constraint identified.",
                actions=[], target_steps=[],
            )

        p_num = primary.step_number
        upstream   = [s for s in steps if int(s.get("step_number", 0)) < p_num]
        downstream = [s for s in steps if int(s.get("step_number", 0)) > p_num]

        actions = []

        if upstream:
            last_upstream = upstream[-1]
            actions.append(TOCAction(
                action=(
                    f"Pace Step {last_upstream.get('step_number')} "
                    f"('{last_upstream.get('title')}') to produce output only at the rate "
                    f"that Step {p_num} can consume. Avoid building a work-in-progress queue."
                ),
                owner=last_upstream.get("actor", "Process Owner"),
                impact="high", effort="medium", automation_type="workflow",
            ))

        if downstream:
            first_downstream = downstream[0]
            actions.append(TOCAction(
                action=(
                    f"Ensure Step {first_downstream.get('step_number')} "
                    f"('{first_downstream.get('title')}') is always ready to pull output "
                    f"from Step {p_num} immediately — eliminate downstream idle time."
                ),
                owner=first_downstream.get("actor", "Process Owner"),
                impact="medium", effort="low", automation_type="workflow",
            ))

        # Suggest automation for high-potential non-constraint steps (free up human capacity)
        high_auto_others = [
            s for s in steps
            if int(s.get("step_number", 0)) != p_num
            and s.get("automation_potential", 0) >= 70
        ]
        for s in high_auto_others[:2]:
            matching_sug = next(
                (sg for sg in suggestions
                 if int(sg.get("step_number", -1)) == int(s.get("step_number", 0))),
                None
            )
            desc = matching_sug.get("description", "Automate this step") if matching_sug else \
                   "Automate this step to redirect freed capacity toward the bottleneck"
            actions.append(TOCAction(
                action=(
                    f"Automate Step {s.get('step_number')} ('{s.get('title')}') "
                    f"[{s.get('automation_potential')}% automation potential]: {desc}. "
                    "This frees human capacity to focus on the bottleneck."
                ),
                owner=s.get("actor", "Process Owner"),
                impact="high", effort="medium",
                automation_type="ai_agent" if s.get("automation_potential", 0) >= 85 else "rpa",
            ))

        target_steps = (
            [int(s.get("step_number", 0)) for s in upstream[-1:]]
            + [int(s.get("step_number", 0)) for s in downstream[:1]]
            + [int(s.get("step_number", 0)) for s in high_auto_others[:2]]
        )

        return TOCPhaseResult(
            phase=3, name="SUBORDINATE",
            summary=(
                f"Align all {len(steps) - 1} supporting steps to the rhythm of the "
                f"bottleneck at Step {p_num}. Automate {len(high_auto_others)} high-potential "
                "non-bottleneck steps to redirect capacity to the constraint."
            ),
            actions=actions,
            target_steps=list(set(target_steps)),
        )

    def _phase_elevate(
        self,
        primary: Optional[ConstraintStep],
        secondary: List[ConstraintStep],
        suggestions: List[Dict],
    ) -> TOCPhaseResult:
        """
        ELEVATE: invest to break the constraint when exploitation is insufficient.
        This is the strategic/high-effort phase.
        """
        if not primary:
            return TOCPhaseResult(
                phase=4, name="ELEVATE",
                summary="No constraint identified.",
                actions=[], target_steps=[],
            )

        actions = []

        # High-effort suggestions for the primary constraint
        high_effort_sugs = [
            s for s in suggestions
            if int(s.get("step_number", -1)) == primary.step_number
            and s.get("effort_level", "medium") == "high"
        ]

        for sug in high_effort_sugs[:2]:
            actions.append(TOCAction(
                action=(
                    f"Strategic investment: {sug.get('title')}. "
                    f"{sug.get('description', '')} — "
                    f"Technologies: {', '.join(sug.get('technologies', []) or ['TBD'])}."
                ),
                owner=primary.actor,
                impact="high", effort="high",
                automation_type=sug.get("automation_type"),
            ))

        # If no high-effort suggestions, generate generic elevation actions
        if not actions:
            actions.append(TOCAction(
                action=(
                    f"Re-engineer Step {primary.step_number} — '{primary.title}': "
                    "consider full process redesign, outsourcing, or technology replacement "
                    "to permanently remove this bottleneck."
                ),
                owner="Executive Sponsor",
                impact="high", effort="high", automation_type="ai_agent",
            ))

        actions.append(TOCAction(
            action=(
                "Implement real-time monitoring on this step (SLA alerts, throughput dashboards) "
                "so future regressions are caught immediately."
            ),
            owner="Operations Manager",
            impact="medium", effort="medium", automation_type="workflow",
        ))

        # Address secondary constraints proactively
        for sc in secondary[:1]:
            actions.append(TOCAction(
                action=(
                    f"Begin preparation to address next constraint: "
                    f"Step {sc.step_number} — '{sc.title}'. "
                    "Start collecting metrics now so Phase 5 (REPEAT) can begin immediately."
                ),
                owner=sc.actor,
                impact="medium", effort="low", automation_type=None,
            ))

        return TOCPhaseResult(
            phase=4, name="ELEVATE",
            summary=(
                f"Break the constraint at Step {primary.step_number} through strategic investment. "
                "Once resolved, the system's throughput ceiling will shift to the next bottleneck."
            ),
            actions=actions,
            target_steps=[primary.step_number] + [s.step_number for s in secondary[:1]],
        )

    def _phase_repeat(self, secondary: List[ConstraintStep]) -> TOCPhaseResult:
        """
        REPEAT: once the primary constraint is resolved, find the next one.
        """
        if not secondary:
            return TOCPhaseResult(
                phase=5, name="REPEAT",
                summary=(
                    "No secondary constraints identified. Re-run TOC analysis after "
                    "implementing Phase 4 changes to discover the next system bottleneck."
                ),
                actions=[TOCAction(
                    action="Re-upload updated process documentation after implementing improvements.",
                    owner="Process Owner",
                    impact="high", effort="low", automation_type=None,
                )],
                target_steps=[],
            )

        next_primary = secondary[0]
        actions = [
            TOCAction(
                action=(
                    f"Promote Step {next_primary.step_number} — '{next_primary.title}' — "
                    "to primary constraint status and restart the 5-step TOC cycle."
                ),
                owner="Process Owner",
                impact="high", effort="low", automation_type=None,
            ),
            TOCAction(
                action=(
                    "Re-run the full analysis pipeline with updated process documentation "
                    "to confirm the new constraint ranking."
                ),
                owner="Operations Manager",
                impact="high", effort="low", automation_type=None,
            ),
        ]

        return TOCPhaseResult(
            phase=5, name="REPEAT",
            summary=(
                f"After resolving the primary constraint, the bottleneck shifts to "
                f"Step {next_primary.step_number} — '{next_primary.title}' "
                f"(bottleneck score: {round(next_primary.bottleneck_score * 100, 1)}%). "
                "Restart the TOC cycle."
            ),
            actions=actions,
            target_steps=[s.step_number for s in secondary],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation (for API / DB storage)
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self, result: TOCAnalysisResult) -> Dict[str, Any]:
        """Convert TOCAnalysisResult to a JSON-serializable dict."""
        def constraint_dict(c: ConstraintStep) -> Dict:
            return {
                "step_number":        c.step_number,
                "title":              c.title,
                "actor":              c.actor,
                "automation_potential": c.automation_potential,
                "pain_points":        c.pain_points,
                "duration_estimate":  c.duration_estimate,
                "step_type":          c.step_type,
                "bottleneck_score":   round(c.bottleneck_score * 100, 1),
                "bottleneck_reasons": c.bottleneck_reasons,
            }

        def action_dict(a: TOCAction) -> Dict:
            return {
                "action":           a.action,
                "owner":            a.owner,
                "impact":           a.impact,
                "effort":           a.effort,
                "automation_type":  a.automation_type,
            }

        def phase_dict(p: TOCPhaseResult) -> Dict:
            return {
                "phase":        p.phase,
                "name":         p.name,
                "summary":      p.summary,
                "actions":      [action_dict(a) for a in p.actions],
                "target_steps": p.target_steps,
            }

        return {
            "process_key":              result.process_key,
            "process_title":            result.process_title,
            "primary_constraint":       constraint_dict(result.primary_constraint)
                                        if result.primary_constraint else None,
            "secondary_constraints":    [constraint_dict(c) for c in result.secondary_constraints],
            "throughput_score":         result.throughput_score,
            "bottleneck_index":         result.bottleneck_index,
            "improvement_potential_pct": result.improvement_potential_pct,
            "summary":                  result.summary,
            "phases":                   [phase_dict(p) for p in result.phases],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_summary(
        self,
        process_title: str,
        primary: Optional[ConstraintStep],
        throughput_score: float,
        improvement_potential: float,
    ) -> str:
        if not primary:
            return (
                f"TOC analysis of '{process_title}': no dominant bottleneck detected. "
                f"Average automation potential is {throughput_score}%."
            )
        return (
            f"TOC analysis of '{process_title}': primary constraint is "
            f"Step {primary.step_number} — '{primary.title}' (actor: {primary.actor}). "
            f"Resolving this bottleneck could improve process throughput by up to "
            f"{improvement_potential}%. Overall automation potential: {throughput_score}%."
        )

    @staticmethod
    def _parse_duration_hours(duration_str: Optional[str]) -> float:
        """Parse '2-4 hours', '1 day', '30 minutes' → float hours."""
        if not duration_str:
            return 0.0
        import re
        s = duration_str.lower()
        nums = re.findall(r"\d+\.?\d*", s)
        if not nums:
            return 0.0
        val = float(nums[0])
        if "day" in s:
            return val * 8
        if "week" in s:
            return val * 40
        if "min" in s:
            return val / 60
        return val  # assume hours

    @staticmethod
    def _max_duration_hours(steps: List[Dict]) -> float:
        from app.core.toc_analyzer import TOCAnalyzer  # avoid circular if called statically
        vals = [TOCAnalyzer._parse_duration_hours(s.get("duration_estimate")) for s in steps]
        return max(vals) if vals else 1.0

    @staticmethod
    def _empty_result(process_key: str, process_title: str) -> TOCAnalysisResult:
        return TOCAnalysisResult(
            process_key=process_key,
            process_title=process_title,
            primary_constraint=None,
            secondary_constraints=[],
            throughput_score=0.0,
            bottleneck_index=0.0,
            phases=[],
            summary="No steps available for TOC analysis.",
            improvement_potential_pct=0.0,
        )

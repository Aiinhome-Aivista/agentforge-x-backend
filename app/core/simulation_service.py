import random

class SimulationService:

    def run_simulation(self, steps, suggestions):
        """
        Simulate agent execution across workflow
        """

        results = []
        total_initial = 0
        total_remaining = 0

        for step in steps:
            step_id = step.get("step_number")
            base_score = step.get("automation_potential", 0)

            total_initial += base_score

            # Find if any agent suggestion exists for this step
            agent = next(
                (s for s in suggestions if s.get("step_key") == step.get("_key")),
                None
            )

            if agent:
                # simulate execution efficiency
                efficiency = self._simulate_agent_efficiency(agent)

                reduced = int(base_score * (efficiency / 100))
                remaining = max(base_score - reduced, 0)

                status = "automated" if remaining < 20 else "partial"
            else:
                remaining = base_score
                status = "manual"

            total_remaining += remaining

            results.append({
                "step_number": step_id,
                "title": step.get("title", ""),
                "initial_score": base_score,
                "remaining_score": remaining,
                "status": status
            })

        return {
            "steps": results,
            "total_initial": total_initial,
            "total_remaining": total_remaining,
            "automation_improvement": round(
                (total_initial - total_remaining) / total_initial * 100, 2
            ) if total_initial else 0
        }


    def _simulate_agent_efficiency(self, agent):
        """
        Simulate variability in agent performance
        """
        base = agent.get("accuracy_estimate", 80)

        # randomness to simulate real-world uncertainty
        variation = random.randint(-10, 5)

        return max(min(base + variation, 100), 50)
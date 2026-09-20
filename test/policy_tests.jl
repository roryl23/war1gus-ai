function representative_state(; gold=500, wood=500, supply=20, demand=4, workers=8,
  barracks=1, lumber_mills=1, blacksmiths=1, stables=1,
  soldiers=2, shooters=2, cavalry=2, catapults=1)
  return UInt32[
    1, 0, 0, 1_000, gold, wood, supply, demand, workers, 1,
    barracks, lumber_mills, blacksmiths, stables, soldiers, shooters, cavalry, catapults,
  ]
end

@testset "state-conditioned transformer policy" begin
  policy = War1gusAI.create_policy()
  early = representative_state(workers=2)
  supply_pressure = representative_state(supply=10, demand=9)
  missing_infrastructure = representative_state(barracks=0, lumber_mills=0, stables=0)
  army_building = representative_state(soldiers=2, shooters=1, cavalry=0, catapults=0)
  blacksmith_building = representative_state(blacksmiths=0, cavalry=0, catapults=0)
  strategic_choice = representative_state(cavalry=0, catapults=0)
  ready_to_attack = representative_state()
  barracks_stage = representative_state(barracks=0, lumber_mills=0, blacksmiths=0, stables=0)
  lumber_stage = representative_state(barracks=1, lumber_mills=0, blacksmiths=0, stables=0)
  blacksmith_stage = representative_state(barracks=1, lumber_mills=1, blacksmiths=0, stables=0)
  stable_stage = representative_state(barracks=1, lumber_mills=1, blacksmiths=1, stables=0)

  progression_actions = War1gusAI.select_action.(Ref(policy), [
    early,
    supply_pressure,
    missing_infrastructure,
    army_building,
    blacksmith_building,
  ])
  @test progression_actions == [0, 1, 2, 4, 3]
  dependency_actions = War1gusAI.select_action.(Ref(policy), [
    barracks_stage,
    lumber_stage,
    blacksmith_stage,
    stable_stage,
  ])
  @test dependency_actions == [2, 2, 3, 2]

  strategic_action = War1gusAI.select_action(policy, strategic_choice)
  prior_only_action = argmax(War1gusAI.action_priors(strategic_choice)) - 1
  @test prior_only_action == 5
  @test strategic_action == 6
  @test strategic_action != prior_only_action
  @test War1gusAI.select_action(policy, ready_to_attack) == 7

  actions = [progression_actions; strategic_action; War1gusAI.select_action(policy, ready_to_attack)]
  @test all(action -> 0 <= action < 10, actions)
  @test all(abs.(War1gusAI.action_logits(policy, strategic_choice)) .<= 1.0f0)
  @test War1gusAI.action_logits(policy, early) != War1gusAI.action_logits(policy, ready_to_attack)
  @test_throws ArgumentError War1gusAI.select_action(policy, early[1:end-1])
end

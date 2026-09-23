-- Noninteractive rollout configuration loaded by Stratagus with -c.
-- All match inputs are supplied by the rollout environment.

local rollout_map = os.getenv("WAR1GUS_ROLLOUT_MAP")
local rollout_seed = tonumber(os.getenv("WAR1GUS_ROLLOUT_SEED")) or 0
local rollout_player = tonumber(os.getenv("WAR1GUS_ROLLOUT_TRAIN_PLAYER")) or 0
local rollout_timeout = tonumber(os.getenv("WAR1GUS_ROLLOUT_TIMEOUT_CYCLES")) or 1
local rollout_match_id = os.getenv("WAR1GUS_ROLLOUT_MATCH_ID") or ""
local rollout_mode = os.getenv("WAR1GUS_ROLLOUT_MODE") or ""

if rollout_timeout < 1 then
  rollout_timeout = 1
end

local function json_string(value)
  value = tostring(value or "")
  value = string.gsub(value, '[%c\\"]', function(character)
    if character == "\"" then
      return "\\\""
    elseif character == "\\" then
      return "\\\\"
    elseif character == "\b" then
      return "\\b"
    elseif character == "\f" then
      return "\\f"
    elseif character == "\n" then
      return "\\n"
    elseif character == "\r" then
      return "\\r"
    elseif character == "\t" then
      return "\\t"
    end
    return string.format("\\u%04x", string.byte(character))
  end)
  return "\"" .. value .. "\""
end

local function json_number(value)
  return tostring(tonumber(value) or 0)
end

local function record_prefix(record_type)
  return "{\"type\":" .. json_string(record_type)
    .. ",\"match_id\":" .. json_string(rollout_match_id)
    .. ",\"map\":" .. json_string(rollout_map)
    .. ",\"mode\":" .. json_string(rollout_mode)
    .. ",\"trainable_player\":" .. json_number(rollout_player)
    .. ",\"seed\":" .. json_number(rollout_seed)
end

local function write_start()
  print(record_prefix("rollout_start") .. "}")
end

local terminal_emitted = false
local original_stop_game
local had_opponents = false

local function write_terminal(outcome)
  local player = rollout_player
  local surviving_total = GetPlayerData(player, "TotalNumUnits")
  local surviving_buildings = GetPlayerData(player, "NumBuildings")
  local surviving_units = math.max(surviving_total - surviving_buildings, 0)
  print(record_prefix("rollout_terminal")
    .. ",\"cycles\":" .. json_number(GameCycle)
    .. ",\"kills\":" .. json_number(GetPlayerData(player, "TotalKills"))
    .. ",\"razings\":" .. json_number(GetPlayerData(player, "TotalRazings"))
    .. ",\"gold\":" .. json_number(GetPlayerData(player, "TotalResources", "gold"))
    .. ",\"wood\":" .. json_number(GetPlayerData(player, "TotalResources", "wood"))
    .. ",\"units\":" .. json_number(surviving_units)
    .. ",\"buildings\":" .. json_number(surviving_buildings)
    .. ",\"total_units\":" .. json_number(GetPlayerData(player, "TotalUnits"))
    .. ",\"total_buildings\":" .. json_number(GetPlayerData(player, "TotalBuildings"))
    .. ",\"outcome\":" .. json_string(outcome)
    .. "}")
end

local function finish(outcome)
  if terminal_emitted then
    return false
  end

  terminal_emitted = true
  write_terminal(outcome)
  if outcome == "win" then
    original_stop_game(GameVictory)
  elseif outcome == "loss" then
    original_stop_game(GameDefeat)
  else
    original_stop_game(GameDraw)
  end
  return false
end

CustomStartup = function()
  InitGameSettings()
  math.randomseed(rollout_seed)

  original_stop_game = StopGame
  ActionVictory = function()
    return had_opponents and finish("win") or false
  end
  ActionDefeat = function()
    return finish("loss")
  end
  ActionDraw = function()
    return finish("draw")
  end

  MapLoadedFuncs:add(function()
    SetGameSpeed(75)
    local fast_forward_enabled = false

    AddTrigger(
      function()
        return not terminal_emitted and GetPlayerData(rollout_player, "TotalNumUnits") == 0
      end,
      function()
        return finish("loss")
      end)
    AddTrigger(
      function()
        local opponents = GetNumOpponents(rollout_player)
        if opponents > 0 then
          had_opponents = true
        end
        return not terminal_emitted and had_opponents and opponents == 0
      end,
      function()
        return finish("win")
      end)
    AddTrigger(
      function()
        -- CreateGame resets fast-forward after MapLoaded; enable it here instead.
        if not fast_forward_enabled then
          SetFastForwardCycle(rollout_timeout)
          fast_forward_enabled = true
        end
        return not terminal_emitted and GameCycle >= rollout_timeout
      end,
      function()
        return finish("timeout")
      end)
  end)

  write_start()
  if DefaultObjectives ~= nil then
    Objectives = DefaultObjectives
  end
  InitGameVariables()
  StartMap(rollout_map)
  Exit(0)
end

Load("scripts/stratagus.lua")
SetTitleScreens({})

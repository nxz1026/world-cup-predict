#!/usr/bin/env python3
"""
World Cup Predict v3.0 — Onside 4-Signal Model + Dixon-Coles + Monte Carlo
P0: form, records, details(ML), spread line 全字段利用
P2: 隐含概率计算 + 加权评分 + 自动校准 + Dixon-Coles 比分分布
P3: Onside 4 信号模型 + 蒙特卡洛冠军模拟 + 通用联赛支持

v3.0 改进（2026-06-30）：
- Onside 4 信号模型：FIFA排名 + 联赛footprint + 东道主 + 足联实力
- Dixon-Coles 双变量泊松（rho=0.2 修正低比分）
- 蒙特卡洛冠军模拟（10k次）
- 通用联赛支持（ESPN / football-data.org / API-Football）
- 世界杯模式 vs 联赛模式

用法: python3 predict_wc.py [--dates YYYYMMDD-YYYYMMDD] [--no-fetch] [--cleanup]
      --no-fetch: 使用本地 /tmp/espn_wc.json (调试用)
      --cleanup: 清理超过7天的 predictions/ 和 results/ 文件
      --monte-carlo: 启用蒙特卡洛冠军模拟
      --league=wc|epl|laliga|bundesliga|seriea|ligue1: 联赛选择
      --data-source=espn|football-data|api-football: 数据源选择
      --n-simulations=10000: 蒙特卡洛模拟次数
输出: JSON 写入 predictions/prediction_YYYY-MM-DD_HH.json
"""
import json, urllib.request, gzip, os, sys, time, math, random
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── 配置 ──────────────────────────────────────
_SKILL_DIR = Path(__file__).parent.parent
FOOTBALL_DIR = Path(os.environ.get("WC_OUTPUT_DIR", str(_SKILL_DIR)))
PREDICTIONS_DIR = FOOTBALL_DIR / "predictions"
RESULTS_DIR = FOOTBALL_DIR / "results"
TRENDS_FILE = _SKILL_DIR / "references" / "tournament-trends.md"
ESPN_URL_TEMPLATE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={dates}&limit=50"

# ─── 重试配置 ───────────────────────────────────
ESPN_MAX_RETRIES = 3
ESPN_RETRY_DELAY_SECONDS = 30
ESPN_TIMEOUT_SECONDS = 15

# ── Dixon-Coles 参数 ────────────────────────────
DC_RHO = 0.2  # 国际足球典型相关系数

# ── 蒙特卡洛参数 ────────────────────────────────
DEFAULT_N_SIMULATIONS = 10000

# ── Onside 4 信号权重 ──────────────────────────
# 市场信号 20% + 4个Onside信号 80%
ONSIDE_WEIGHTS = {
    "market_odds":     0.20,  # 盘口隐含概率（市场信号）
    "fifa_ranking":    0.25,  # FIFA排名分
    "league_footprint": 0.20,  # 联赛球员占比/分档
    "host_advantage":  0.15,  # 东道主优势
    "confederation":   0.20,  # 足联实力
}

# ── 足联实力系数 ──────────────────────────────
CONFEDERATION_STRENGTH = {
    "UEFA":    1.00,   # 欧洲
    "CONMEBOL": 0.95,  # 南美
    "CONCACAF": 0.70,  # 中北美
    "AFC":     0.65,   # 亚洲
    "CAF":     0.60,   # 非洲
    "OFC":     0.40,   # 大洋洲
}

# ── 国家→足联映射（常用） ──────────────────────
COUNTRY_CONFEDERATION = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA",
    "Italy": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Denmark": "UEFA", "Poland": "UEFA",
    "Austria": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA", "Wales": "UEFA",
    "Turkey": "UEFA", "Norway": "UEFA", "Sweden": "UEFA", "Ukraine": "UEFA",
    "Czech Republic": "UEFA", "Czechia": "UEFA", "Hungary": "UEFA", "Greece": "UEFA",
    "Romania": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA", "Finland": "UEFA",
    "Ireland": "UEFA", "Iceland": "UEFA", "Northern Ireland": "UEFA",
    "USA": "CONCACAF", "United States": "CONCACAF", "Mexico": "CONCACAF",
    "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Panama": "CONCACAF",
    "Jamaica": "CONCACAF", "Honduras": "CONCACAF",
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC",
    "Australia": "AFC", "Iran": "AFC", "Saudi Arabia": "AFC",
    "Qatar": "AFC", "China PR": "AFC", "China": "AFC",
    "Iraq": "AFC", "United Arab Emirates": "AFC", "Uzbekistan": "AFC",
    "Nigeria": "CAF", "Egypt": "CAF", "Senegal": "CAF", "Morocco": "CAF",
    "Cameroon": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF", "Cote d'Ivoire": "CAF",
    "Algeria": "CAF", "Tunisia": "CAF", "Mali": "CAF", "South Africa": "CAF",
    "DR Congo": "CAF", "Congo DR": "CAF",
    "New Zealand": "OFC",
}

# ── 联赛配置 ────────────────────────────────────
LEAGUE_CONFIG = {
    "wc": {
        "name": "FIFA World Cup",
        "tournament_type": "world_cup",
        "data_source": "espn",
        "league_id": "fifa.world",
        "host_country": None,  # 动态设置
        "groups": True,
        "knockout": True,
    },
    "epl": {
        "name": "English Premier League",
        "tournament_type": "league",
        "data_source": "football-data",
        "league_id": "PL",
        "host_country": "England",
        "groups": False,
        "knockout": False,
    },
    "laliga": {
        "name": "La Liga",
        "tournament_type": "league",
        "data_source": "football-data",
        "league_id": "PD",
        "host_country": "Spain",
        "groups": False,
        "knockout": False,
    },
    "bundesliga": {
        "name": "Bundesliga",
        "tournament_type": "league",
        "data_source": "football-data",
        "league_id": "BL1",
        "host_country": "Germany",
        "groups": False,
        "knockout": False,
    },
    "seriea": {
        "name": "Serie A",
        "tournament_type": "league",
        "data_source": "football-data",
        "league_id": "SA",
        "host_country": "Italy",
        "groups": False,
        "knockout": False,
    },
    "ligue1": {
        "name": "Ligue 1",
        "tournament_type": "league",
        "data_source": "football-data",
        "league_id": "FL1",
        "host_country": "France",
        "groups": False,
        "knockout": False,
    },
}

# ── 东道主加成 ──────────────────────────────────
HOST_ADVANTAGE_BOOST = 0.05  # 固定加成到 home_strength


def log(msg):
    print(f"[predict] {msg}", file=sys.stderr)


# ─── 累积校准（回填→预测反馈） ─────────────────
def load_historical_past_matches(days=30):
    """读取历史 predictions/ 文件中的 past_matches + references/historical_past_matches.json，去重后返回"""
    all_past = []
    cutoff = time.time() - days * 86400
    
    # 从 predictions/ 窗口文件读取
    for f in PREDICTIONS_DIR.glob("prediction_*.json"):
        if f.stat().st_mtime < cutoff:
            continue
        try:
            d = json.load(open(f))
            all_past.extend(d.get("past_matches", []))
        except Exception:
            pass
    
    # 从 references/historical_past_matches.json 读取（预填充的历史数据）
    hist_file = FOOTBALL_DIR / "references" / "historical_past_matches.json"
    if hist_file.exists():
        try:
            hist_data = json.load(open(hist_file))
            all_past.extend(hist_data)
        except Exception:
            pass
    
    # 去重（同一场比赛可能在多个窗口中出现）
    seen = set()
    unique = []
    for m in all_past:
        key = f"{m.get('kickoff_utc','')}_{m.get('home','')}_{m.get('away','')}"
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def compute_calibration_offset(past_matches):
    """
    从累积 past_matches 计算 calibration 修正因子。
    用实际赛果分布 vs 均匀分布(1/3)的比率做软修正。
    返回 dict 或 None（样本不足时）。
    """
    if len(past_matches) < 5:
        return None

    home_wins = draws = away_wins = 0

    for m in past_matches:
        score = m.get("score", "")
        if not score or "-" not in score:
            continue
        parts = score.split("-")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        hs, aws = int(parts[0]), int(parts[1])
        if hs > aws:
            home_wins += 1
        elif hs == aws:
            draws += 1
        else:
            away_wins += 1

    total = home_wins + draws + away_wins
    if total < 5:
        return None

    actual_home_rate = home_wins / total
    actual_draw_rate = draws / total
    actual_away_rate = away_wins / total

    # 修正因子：实际分布 vs 均匀分布(1/3)的比率
    # 限制在 [0.5, 2.0] 避免极端
    home_correction = max(0.5, min(2.0, actual_home_rate / (1/3)))
    draw_correction = max(0.5, min(2.0, actual_draw_rate / (1/3)))
    away_correction = max(0.5, min(2.0, actual_away_rate / (1/3)))

    return {
        "home_correction": round(home_correction, 3),
        "draw_correction": round(draw_correction, 3),
        "away_correction": round(away_correction, 3),
        "sample_size": total,
        "actual_home_rate": round(actual_home_rate, 3),
        "actual_draw_rate": round(actual_draw_rate, 3),
        "actual_away_rate": round(actual_away_rate, 3),
    }


# ─── 存储清理（保留7天） ────────────────────────
def cleanup_old_files(days=7):
    """清理超过 N 天的 predictions/ 和 results/ 文件"""
    cutoff = time.time() - days * 86400
    removed = 0
    for directory in [PREDICTIONS_DIR, RESULTS_DIR]:
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
                log(f"Cleaned up: {f.name}")
    if removed > 0:
        log(f"Cleanup complete: removed {removed} files older than {days} days")
    return removed


# ── ML 解析 ────────────────────────────────────
def parse_american_odds(odds_str):
    """解析美式赔率 → 隐含概率 (含 vig)"""
    try:
        raw = str(odds_str).strip().lstrip('+')
        odds = int(raw)
        abs_odds = abs(odds)
        if odds < 0:
            return abs_odds / (abs_odds + 100)
        else:
            return 100 / (abs_odds + 100)
    except (ValueError, TypeError):
        return None


def parse_details(details_str):
    """解析 details 字段如 'CZE -125' → (team, odds_str, implied)"""
    if not details_str:
        return None, None, None
    parts = details_str.strip().split()
    if len(parts) >= 2:
        team = parts[0]
        odds_str = parts[-1]
        impl = parse_american_odds(odds_str)
        return team, odds_str, impl
    return None, None, None


# ── 球队状态评分 ──────────────────────────────
COUNTRY_CN = {
    "Afghanistan": "阿富汗", "Albania": "阿尔巴尼亚", "Algeria": "阿尔及利亚",
    "Angola": "安哥拉", "Argentina": "阿根廷", "Armenia": "亚美尼亚",
    "Australia": "澳大利亚", "Austria": "奥地利", "Azerbaijan": "阿塞拜疆",
    "Bahrain": "巴林", "Bangladesh": "孟加拉国", "Belarus": "白俄罗斯",
    "Belgium": "比利时", "Benin": "贝宁", "Bolivia": "玻利维亚",
    "Bosnia and Herzegovina": "波黑", "Bosnia-Herzegovina": "波黑",
    "Botswana": "博茨瓦纳", "Brazil": "巴西", "Bulgaria": "保加利亚",
    "Burkina Faso": "布基纳法索", "Burundi": "布隆迪", "Cameroon": "喀麦隆",
    "Canada": "加拿大", "Cape Verde": "佛得角", "Chad": "乍得",
    "Chile": "智利", "China PR": "中国", "China": "中国",
    "Colombia": "哥伦比亚", "Comoros": "科摩罗", "Congo": "刚果",
    "Congo DR": "刚果(金)", "DR Congo": "刚果(金)", "Costa Rica": "哥斯达黎加",
    "Croatia": "克罗地亚", "Cuba": "古巴", "Curacao": "库拉索",
    "Curaçao": "库拉索", "Cyprus": "塞浦路斯", "Czechia": "捷克",
    "Czech Republic": "捷克", "Denmark": "丹麦", "Djibouti": "吉布提",
    "Dominican Republic": "多米尼加", "Ecuador": "厄瓜多尔", "Egypt": "埃及",
    "El Salvador": "萨尔瓦多", "England": "英格兰", "Estonia": "爱沙尼亚",
    "Eswatini": "斯威士兰", "Ethiopia": "埃塞俄比亚", "Faroe Islands": "法罗群岛",
    "Fiji": "斐济", "Finland": "芬兰", "France": "法国",
    "Gabon": "加蓬", "Gambia": "冈比亚", "Georgia": "格鲁吉亚",
    "Germany": "德国", "Ghana": "加纳", "Gibraltar": "直布罗陀",
    "Greece": "希腊", "Grenada": "格林纳达", "Guadeloupe": "瓜德罗普",
    "Guatemala": "危地马拉", "Guinea": "几内亚", "Guinea-Bissau": "几内亚比绍",
    "Guyana": "圭亚那", "Haiti": "海地", "Honduras": "洪都拉斯",
    "Hong Kong": "中国香港", "Hungary": "匈牙利", "Iceland": "冰岛",
    "India": "印度", "Indonesia": "印度尼西亚", "Iran": "伊朗",
    "Iraq": "伊拉克", "Ireland": "爱尔兰", "Israel": "以色列",
    "Italy": "意大利", "Ivory Coast": "科特迪瓦", "Cote d'Ivoire": "科特迪瓦",
    "Jamaica": "牙买加", "Japan": "日本", "Jordan": "约旦",
    "Kazakhstan": "哈萨克斯坦", "Kenya": "肯尼亚", "Korea Republic": "韩国",
    "South Korea": "韩国", "Korea DPR": "朝鲜", "North Korea": "朝鲜",
    "Kosovo": "科索沃", "Kuwait": "科威特", "Kyrgyzstan": "吉尔吉斯斯坦",
    "Laos": "老挝", "Latvia": "拉脱维亚", "Lebanon": "黎巴嫩",
    "Lesotho": "莱索托", "Liberia": "利比里亚", "Libya": "利比亚",
    "Liechtenstein": "列支敦士登", "Lithuania": "立陶宛", "Luxembourg": "卢森堡",
    "Macao": "中国澳门", "Macedonia": "北马其顿", "North Macedonia": "北马其顿",
    "Madagascar": "马达加斯加", "Malawi": "马拉维", "Malaysia": "马来西亚",
    "Maldives": "马尔代夫", "Mali": "马里", "Malta": "马耳他",
    "Martinique": "马提尼克", "Mauritania": "毛里塔尼亚", "Mauritius": "毛里求斯",
    "Mexico": "墨西哥", "Moldova": "摩尔多瓦", "Monaco": "摩纳哥",
    "Mongolia": "蒙古", "Montenegro": "黑山", "Morocco": "摩洛哥",
    "Mozambique": "莫桑比克", "Myanmar": "缅甸", "Namibia": "纳米比亚",
    "Nepal": "尼泊尔", "Netherlands": "荷兰", "New Caledonia": "新喀里多尼亚",
    "New Zealand": "新西兰", "Nicaragua": "尼加拉瓜", "Niger": "尼日尔",
    "Nigeria": "尼日利亚", "Norway": "挪威", "Oman": "阿曼",
    "Pakistan": "巴基斯坦", "Palestine": "巴勒斯坦", "Panama": "巴拿马",
    "Paraguay": "巴拉圭", "Peru": "秘鲁", "Philippines": "菲律宾",
    "Poland": "波兰", "Portugal": "葡萄牙", "Qatar": "卡塔尔",
    "Romania": "罗马尼亚", "Russia": "俄罗斯", "Rwanda": "卢旺达",
    "Saudi Arabia": "沙特", "Scotland": "苏格兰", "Senegal": "塞内加尔",
    "Serbia": "塞尔维亚", "Sierra Leone": "塞拉利昂", "Singapore": "新加坡",
    "Slovakia": "斯洛伐克", "Slovenia": "斯洛文尼亚", "Solomon Islands": "所罗门群岛",
    "Somalia": "索马里", "South Africa": "南非", "South Sudan": "南苏丹",
    "Spain": "西班牙", "Sri Lanka": "斯里兰卡", "Sudan": "苏丹",
    "Suriname": "苏里南", "Sweden": "瑞典", "Switzerland": "瑞士",
    "Syria": "叙利亚", "Tahiti": "塔希提", "Taiwan": "中国台北",
    "Tajikistan": "塔吉克斯坦", "Tanzania": "坦桑尼亚", "Thailand": "泰国",
    "Togo": "多哥", "Trinidad and Tobago": "特立尼达和多巴哥",
    "Tunisia": "突尼斯", "Turkey": "土耳其", "Türkiye": "土耳其",
    "Turkmenistan": "土库曼斯坦", "Uganda": "乌干达", "Ukraine": "乌克兰",
    "United Arab Emirates": "阿联酋", "Uruguay": "乌拉圭",
    "United States": "美国", "USA": "美国", "Uzbekistan": "乌兹别克斯坦",
    "Venezuela": "委内瑞拉", "Vietnam": "越南", "Wales": "威尔士",
    "Yemen": "也门", "Zambia": "赞比亚", "Zimbabwe": "津巴布韦",
    # 俱乐部中文
    "Manchester City": "曼城", "Manchester United": "曼联", "Liverpool": "利物浦",
    "Chelsea": "切尔西", "Arsenal": "阿森纳", "Tottenham": "热刺",
    "Newcastle": "纽卡斯尔", "Aston Villa": "阿斯顿维拉", "Brighton": "布莱顿",
    "West Ham": "西汉姆", "Crystal Palace": "水晶宫", "Wolverhampton": "狼队",
    "Fulham": "富勒姆", "Bournemouth": "伯恩茅斯", "Nottingham Forest": "诺丁汉森林",
    "Brentford": "布伦特福德", "Everton": "埃弗顿", "Leicester": "莱斯特城",
    "Ipswich": "伊普斯维奇", "Southampton": "南安普顿",
    "Real Madrid": "皇家马德里", "Barcelona": "巴塞罗那", "Atletico Madrid": "马竞",
    "Sevilla": "塞维利亚", "Real Sociedad": "皇家社会", "Villarreal": "比利亚雷亚尔",
    "Athletic Club": "毕尔巴鄂", "Real Betis": "贝蒂斯",
    "Bayern Munich": "拜仁慕尼黑", "Borussia Dortmund": "多特蒙德",
    "RB Leipzig": "RB莱比锡", "Bayer Leverkusen": "勒沃库森",
    "Wolfsburg": "沃尔夫斯堡", "Frankfurt": "法兰克福",
    "Juventus": "尤文图斯", "AC Milan": "AC米兰", "Inter Milan": "国际米兰",
    "Napoli": "那不勒斯", "Roma": "罗马", "Lazio": "拉齐奥",
    "Atalanta": "亚特兰大", "Fiorentina": "佛罗伦萨",
    "Paris Saint-Germain": "巴黎圣日耳曼", "PSG": "巴黎圣日耳曼",
    "Marseille": "马赛", "Lyon": "里昂", "Monaco": "摩纳哥",
    "Lille": "里尔", "Nice": "尼斯",
}


def to_cn(name):
    """英文国家名/俱乐部名 → 中文"""
    if not name:
        return name
    return COUNTRY_CN.get(name, COUNTRY_CN.get(name.replace("'", ""), name))


def form_to_score(form_str):
    """'DWDDW' → 0-1, W=3, D=1, L=0"""
    if not form_str:
        return 0.5
    score = sum(3 if c == 'W' else 1 if c == 'D' else 0 for c in form_str)
    return score / (len(form_str) * 3)


def record_to_score(records):
    """records[0].summary '1-0-0' (W-D-L) → 0-1"""
    if not records:
        return 0.5
    summary = records[0].get("summary", "")
    parts = summary.split("-")
    if len(parts) >= 3:
        w, d, l = int(parts[0]), int(parts[1]), int(parts[2])
        total = w + d + l
        return (w * 3 + d) / (total * 3) if total > 0 else 0.5
    return 0.5


# ── 亚盘 movement 分析 ───────────────────────
def spread_movement_factor(away_close):
    """用 away spread close 的 line 判断 market 方向."""
    if not away_close:
        return 0.0
    line = away_close.get("line", None)
    if line is None:
        return 0.0
    try:
        return max(-1.0, min(1.0, float(line) / 3.0))
    except (ValueError, TypeError):
        return 0.0


# ── vig 去除 ──────────────────────────────────
def remove_vig(home_p, draw_p, away_p=None, default_margin=1.07):
    """三向去水"""
    if draw_p is None:
        return None, None, None
    if home_p is None and away_p is None:
        return None, None, None
    if away_p is None:
        away_p = default_margin - home_p - draw_p
        if away_p < 0:
            away_p = 0.05
    if home_p is None:
        home_p = default_margin - draw_p - away_p
        if home_p < 0:
            home_p = 0.05
    total = home_p + draw_p + away_p
    if total <= 0:
        return home_p / default_margin, draw_p / default_margin, away_p / default_margin
    return home_p / total, draw_p / total, away_p / total


# ── Poisson 置信区间 ──────────────────────────
def poisson_confidence_interval(lam, confidence=0.95):
    """
    Poisson 分布的置信区间（Garwood 精确法近似）
    返回 (lower, upper) — 95% CI
    """
    if lam <= 0:
        return (0, 0)
    # 使用正态近似（λ > 10 时效果好，λ < 10 时用查表法简化）
    if lam >= 10:
        z = 1.96  # 95% CI
        lower = max(0, lam - z * math.sqrt(lam))
        upper = lam + z * math.sqrt(lam)
    else:
        # 小 λ 用简化的查表法（基于 Poisson 分布表）
        # 95% CI 近似为 [λ - 1.96√λ, λ + 1.96√λ]，下限不低于 0
        lower = max(0, lam - 1.96 * math.sqrt(lam + 0.5))
        upper = lam + 1.96 * math.sqrt(lam + 0.5)
    return (round(lower, 1), round(upper, 1))


def poisson_pmf(k, lam):
    """Poisson 概率质量函数 P(X=k) = λ^k * e^(-λ) / k!"""
    if k < 0:
        return 0.0
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    log_p = k * math.log(lam) - lam - math.lgamma(k + 1)
    return math.exp(log_p)


# ─── Dixon-Coles 双变量泊松 ────────────────────
def tau_correction(home_goals, away_goals, lambda_h, lambda_a, rho=DC_RHO):
    """
    Dixon-Coles 低比分修正函数。
    
    修正低比分概率（0-0, 1-0, 0-1, 1-1）：
    - τ(0,0) = 1 - ρ
    - τ(1,0) = 1 + ρ * λ_a
    - τ(0,1) = 1 + ρ * λ_h
    - τ(1,1) = 1 - ρ * λ_h * λ_a
    - 其他 = 1.0（无修正）
    
    Args:
        home_goals: 主队进球数
        away_goals: 客队进球数
        lambda_h: 主队期望进球
        lambda_a: 客队期望进球
        rho: 相关系数（默认 0.2）
    
    Returns:
        修正因子 τ
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + rho * lambda_a
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + rho * lambda_h
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho * lambda_h * lambda_a
    else:
        return 1.0


def dixon_coles_pmf(home_goals, away_goals, lambda_h, lambda_a, rho=DC_RHO):
    """
    Dixon-Coles 双变量泊松概率质量函数。
    
    P(H=h, A=a) = Poisson(h, λ_h) * Poisson(a, λ_a) * τ(h, a, λ_h, λ_a, ρ)
    
    Args:
        home_goals: 主队进球数
        away_goals: 客队进球数
        lambda_h: 主队期望进球
        lambda_a: 客队期望进球
        rho: 相关系数
    
    Returns:
        修正后的联合概率
    """
    base_prob = poisson_pmf(home_goals, lambda_h) * poisson_pmf(away_goals, lambda_a)
    tau = tau_correction(home_goals, away_goals, lambda_h, lambda_a, rho)
    return base_prob * tau


def dixon_coles_match_probs(lambda_h, lambda_a, rho=DC_RHO, max_goals=8):
    """
    计算 Dixon-Coles 模型下的胜平负概率。
    
    Args:
        lambda_h: 主队期望进球
        lambda_a: 客队期望进球
        rho: 相关系数
        max_goals: 最大进球数（截断）
    
    Returns:
        dict: {home_win_prob, draw_prob, away_win_prob, score_probs: [(h,a,p),...]}
    """
    score_probs = []
    home_win_p = 0.0
    draw_p = 0.0
    away_win_p = 0.0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = dixon_coles_pmf(h, a, lambda_h, lambda_a, rho)
            if p > 0.0001:
                score_probs.append((h, a, p))
                if h > a:
                    home_win_p += p
                elif h == a:
                    draw_p += p
                else:
                    away_win_p += p
    
    # 归一化（截断可能导致概率和 < 1）
    total = home_win_p + draw_p + away_win_p
    if total > 0:
        home_win_p /= total
        draw_p /= total
        away_win_p /= total
    
    score_probs.sort(key=lambda x: -x[2])
    
    return {
        "home_win": round(home_win_p, 4),
        "draw": round(draw_p, 4),
        "away_win": round(away_win_p, 4),
        "score_probs": score_probs[:12],  # top 12 scores
    }


# ─── Onside 4 信号模型 ─────────────────────────
def fetch_fifa_rankings():
    """获取 FIFA 世界排名，返回 {country_name: rank} 字典"""
    rank_file = FOOTBALL_DIR / "references" / "fifa_rankings.json"
    if rank_file.exists():
        with open(rank_file) as f:
            data = json.load(f)
        # 支持两种格式: 列表 [{country, rank}] 或 字典 {country: rank}
        if isinstance(data, list):
            return {item.get("country", item.get("name", "")): item.get("rank", item.get("fifa_rank", 200)) for item in data}
        elif isinstance(data, dict):
            # 检查是否是 {country: rank} 格式
            first_val = next(iter(data.values())) if data else None
            if isinstance(first_val, (int, float)):
                return data
            # 可能是 {country: {rank: X, points: Y}} 格式
            result = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    result[k] = v.get("rank", 200)
                else:
                    result[k] = v
            return result
    return {}


def fifa_rank_to_score(rank, max_rank=200):
    """
    将 FIFA 排名映射为 0-1 分数。
    排名 1 → 1.0，排名 max_rank → 0.0
    使用指数衰减：score = exp(-3 * (rank-1) / max_rank)
    """
    if not rank or rank <= 0:
        return 0.5
    rank = min(rank, max_rank)
    score = math.exp(-3.0 * (rank - 1) / max_rank)
    return max(0.0, min(1.0, score))


def get_confederation(country_name):
    """获取国家所属足联"""
    return COUNTRY_CONFEDERATION.get(country_name, "UEFA")  # 默认 UEFA


def confederation_score(country_name):
    """
    足联实力评分 0-1。
    UEFA/CONMEBOL 最高，AFC/CAF 中等，CONCACF/OFC 较低。
    """
    conf = get_confederation(country_name)
    return CONFEDERATION_STRENGTH.get(conf, 0.5)


def league_footprint_score(country_name, fifa_rank=None):
    """
    联赛 footprint 评分 0-1。
    用 FIFA 排名分档替代实际球员占比数据：
    - Top 10: 1.0（五大联赛核心球员多）
    - 11-30: 0.8
    - 31-60: 0.6
    - 61-100: 0.4
    - 100+: 0.2
    
    对于俱乐部比赛，直接返回 0.8（默认高联赛水平）
    """
    if fifa_rank and fifa_rank > 0:
        if fifa_rank <= 10:
            return 1.0
        elif fifa_rank <= 30:
            return 0.8
        elif fifa_rank <= 60:
            return 0.6
        elif fifa_rank <= 100:
            return 0.4
        else:
            return 0.2
    # 无 FIFA 排名时，按足联推断
    conf = get_confederation(country_name)
    if conf in ("UEFA", "CONMEBOL"):
        return 0.7
    elif conf in ("AFC", "CAF", "CONCACAF"):
        return 0.5
    else:
        return 0.3


def host_advantage_score(country_name, host_country):
    """
    东道主优势评分。
    如果是东道主 → 1.0 + 固定加成
    否则 → 0.5（中性）
    """
    if host_country and country_name == host_country:
        return 1.0
    return 0.5


def compute_onside_signals(home_team, away_team, fifa_rankings, host_country=None):
    """
    计算 Onside 4 信号评分。
    
    Args:
        home_team: 主队名（英文）
        away_team: 客队名（英文）
        fifa_rankings: FIFA 排名字典 {country: rank}
        host_country: 东道主国家（可选）
    
    Returns:
        dict: 包含 4 个信号分数和综合评分
    """
    home_rank = fifa_rankings.get(home_team, 100)
    away_rank = fifa_rankings.get(away_team, 100)
    
    # 信号1: FIFA 排名分
    home_fifa = fifa_rank_to_score(home_rank)
    away_fifa = fifa_rank_to_score(away_rank)
    
    # 信号2: 联赛 footprint
    home_league = league_footprint_score(home_team, home_rank)
    away_league = league_footprint_score(away_team, away_rank)
    
    # 信号3: 东道主优势
    home_host = host_advantage_score(home_team, host_country)
    away_host = host_advantage_score(away_team, host_country)
    
    # 信号4: 足联实力
    home_conf = confederation_score(home_team)
    away_conf = confederation_score(away_team)
    
    # 综合评分（加权）
    w = ONSIDE_WEIGHTS
    home_onside = (
        home_fifa * w["fifa_ranking"]
        + home_league * w["league_footprint"]
        + home_host * w["host_advantage"]
        + home_conf * w["confederation"]
    )
    away_onside = (
        away_fifa * w["fifa_ranking"]
        + away_league * w["league_footprint"]
        + away_host * w["host_advantage"]
        + away_conf * w["confederation"]
    )
    
    return {
        "home": {
            "fifa_rank": home_rank,
            "fifa_score": round(home_fifa, 3),
            "league_footprint": round(home_league, 3),
            "host_advantage": round(home_host, 3),
            "confederation": round(home_conf, 3),
            "onside_score": round(home_onside, 3),
        },
        "away": {
            "fifa_rank": away_rank,
            "fifa_score": round(away_fifa, 3),
            "league_footprint": round(away_league, 3),
            "host_advantage": round(away_host, 3),
            "confederation": round(away_conf, 3),
            "onside_score": round(away_onside, 3),
        }
    }


# ─── 数据源抽象层 ──────────────────────────────
def fetch_events(dates_str, league_key="wc", data_source="espn"):
    """
    抽象数据源层：根据联赛配置获取比赛数据。
    
    Args:
        dates_str: 日期范围字符串 (YYYYMMDD-YYYYMMDD)
        league_key: 联赛键 (wc, epl, laliga, bundesliga, seriea, ligue1)
        data_source: 数据源 (espn, football-data, api-football)
    
    Returns:
        list: events 列表
    """
    config = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["wc"])
    
    if data_source == "espn" or config["data_source"] == "espn":
        return fetch_espn(dates_str)
    elif data_source == "football-data" or config["data_source"] == "football-data":
        return fetch_football_data(dates_str, config)
    elif data_source == "api-football":
        return fetch_api_football(dates_str, config)
    else:
        # 默认 ESPN
        return fetch_espn(dates_str)


def fetch_espn(dates_str):
    """抓取 ESPN 数据（带重试机制）, 返回 parsed events"""
    url = ESPN_URL_TEMPLATE.format(dates=dates_str)
    
    for attempt in range(1, ESPN_MAX_RETRIES + 1):
        try:
            log(f"Fetching ESPN (attempt {attempt}/{ESPN_MAX_RETRIES}): {url}")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                'Accept-Encoding': 'gzip'
            })
            resp = urllib.request.urlopen(req, timeout=ESPN_TIMEOUT_SECONDS)
            data = json.loads(gzip.decompress(resp.read()))
            
            # 存一份到 /tmp 供调试
            with open("/tmp/espn_wc.json", "w") as f:
                json.dump(data, f, indent=2)
            
            return data.get("events", [])
        
        except Exception as e:
            log(f"Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < ESPN_MAX_RETRIES:
                log(f"Retrying in {ESPN_RETRY_DELAY_SECONDS}s...")
                time.sleep(ESPN_RETRY_DELAY_SECONDS)
            else:
                log(f"All {ESPN_MAX_RETRIES} attempts failed")
                raise


def fetch_football_data(dates_str, config):
    """
    从 football-data.org 获取数据。
    注意：需要 API key（环境变量 FOOTBALL_DATA_API_KEY）
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    league_id = config["league_id"]
    
    # 解析日期范围
    if "-" in dates_str:
        start_date = dates_str.split("-")[0]
        end_date = dates_str.split("-")[1]
    else:
        start_date = dates_str
        end_date = dates_str
    
    url = f"https://api.football-data.org/v4/competitions/{league_id}/matches?dateFrom={start_date}&dateTo={end_date}"
    
    headers = {
        'User-Agent': 'WorldCupPredict/3.0',
    }
    if api_key:
        headers['X-Auth-Token'] = api_key
    
    try:
        log(f"Fetching football-data.org: {url}")
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=ESPN_TIMEOUT_SECONDS)
        data = json.loads(resp.read())
        
        # 转换为 ESPN 格式
        return convert_football_data_to_espn_format(data, config)
    except Exception as e:
        log(f"football-data.org fetch failed: {e}")
        # 降级到 ESPN
        log("Falling back to ESPN...")
        return fetch_espn(dates_str)


def fetch_api_football(dates_str, config):
    """
    从 API-Football 获取数据。
    注意：需要 API key（环境变量 API_FOOTBALL_KEY）
    """
    api_key = os.environ.get("API_FOOTBALL_KEY", "")
    league_id = config["league_id"]
    
    if "-" in dates_str:
        date = dates_str.split("-")[0]
    else:
        date = dates_str
    
    # 转换日期格式 YYYYMMDD → YYYY-MM-DD
    formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season=2024&date={formatted_date}"
    
    headers = {
        'User-Agent': 'WorldCupPredict/3.0',
        'x-apisports-key': api_key,
    }
    
    try:
        log(f"Fetching API-Football: {url}")
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=ESPN_TIMEOUT_SECONDS)
        data = json.loads(resp.read())
        
        return convert_api_football_to_espn_format(data, config)
    except Exception as e:
        log(f"API-Football fetch failed: {e}")
        log("Falling back to ESPN...")
        return fetch_espn(dates_str)


def convert_football_data_to_espn_format(data, config):
    """将 football-data.org 格式转换为 ESPN 兼容格式"""
    events = []
    matches = data.get("matches", [])
    
    for match in matches:
        home_team = match.get("homeTeam", {}).get("name", "Unknown")
        away_team = match.get("awayTeam", {}).get("name", "Unknown")
        
        # 状态映射
        status_map = {
            "SCHEDULED": "STATUS_SCHEDULED",
            "LIVE": "STATUS_IN_PROGRESS",
            "IN_PLAY": "STATUS_IN_PROGRESS",
            "PAUSED": "STATUS_IN_PROGRESS",
            "FINISHED": "STATUS_FULL_TIME",
            "POSTPONED": "STATUS_SCHEDULED",
            "SUSPENDED": "STATUS_SCHEDULED",
            "CANCELLED": "STATUS_SCHEDULED",
        }
        
        status = status_map.get(match.get("status", ""), "STATUS_SCHEDULED")
        completed = match.get("status") == "FINISHED"
        
        # 比分
        score_data = match.get("score", {}).get("fullTime", {})
        home_score = str(score_data.get("home", 0) or 0)
        away_score = str(score_data.get("away", 0) or 0)
        
        event = {
            "name": f"{away_team} at {home_team}",
            "date": match.get("utcDate", ""),
            "competitions": [{
                "status": {
                    "type": {
                        "name": status,
                        "completed": completed,
                    }
                },
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": home_team, "abbreviation": match.get("homeTeam", {}).get("t3", "UNK")},
                        "score": home_score,
                        "form": "",
                        "records": [],
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": away_team, "abbreviation": match.get("awayTeam", {}).get("t3", "UNK")},
                        "score": away_score,
                        "form": "",
                        "records": [],
                    },
                ],
                "odds": [],
            }]
        }
        events.append(event)
    
    return events


def convert_api_football_to_espn_format(data, config):
    """将 API-Football 格式转换为 ESPN 兼容格式"""
    events = []
    fixtures = data.get("response", [])
    
    for fixture in fixtures:
        home_team = fixture.get("teams", {}).get("home", {}).get("name", "Unknown")
        away_team = fixture.get("teams", {}).get("away", {}).get("name", "Unknown")
        
        status_short = fixture.get("fixture", {}).get("status", {}).get("short", "NS")
        status_map = {
            "NS": "STATUS_SCHEDULED",
            "1H": "STATUS_IN_PROGRESS",
            "HT": "STATUS_IN_PROGRESS",
            "2H": "STATUS_IN_PROGRESS",
            "ET": "STATUS_IN_PROGRESS",
            "P": "STATUS_IN_PROGRESS",
            "FT": "STATUS_FULL_TIME",
            "AET": "STATUS_FINAL_ET",
            "PEN": "STATUS_FINAL_PEN",
            "SUSP": "STATUS_SCHEDULED",
            "INT": "STATUS_SCHEDULED",
            "PST": "STATUS_SCHEDULED",
            "CANC": "STATUS_SCHEDULED",
            "ABD": "STATUS_SCHEDULED",
            "AWD": "STATUS_FULL_TIME",
            "WO": "STATUS_FULL_TIME",
        }
        
        status = status_map.get(status_short, "STATUS_SCHEDULED")
        completed = status in ("STATUS_FULL_TIME", "STATUS_FINAL_ET", "STATUS_FINAL_PEN")
        
        goals = fixture.get("goals", {})
        home_score = str(goals.get("home", 0) or 0)
        away_score = str(goals.get("away", 0) or 0)
        
        event = {
            "name": f"{away_team} at {home_team}",
            "date": fixture.get("fixture", {}).get("date", ""),
            "competitions": [{
                "status": {
                    "type": {
                        "name": status,
                        "completed": completed,
                    }
                },
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"displayName": home_team, "abbreviation": home_team[:3].upper()},
                        "score": home_score,
                        "form": "",
                        "records": [],
                    },
                    {
                        "homeAway": "away",
                        "team": {"displayName": away_team, "abbreviation": away_team[:3].upper()},
                        "score": away_score,
                        "form": "",
                        "records": [],
                    },
                ],
                "odds": [],
            }]
        }
        events.append(event)
    
    return events


# ── 事件解析 ──────────────────────────────────
def parse_events(events, now_utc=None):
    """解析 ESPN events → 结束比赛列表 + 待预测比赛列表"""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    
    past = []
    future = []
    in_progress = []
    
    for ev in events:
        en_name = ev.get("name", "")
        if " at " in en_name:
            away_en, home_en = en_name.split(" at ", 1)
            name = f"{to_cn(away_en)} vs {to_cn(home_en)}"
        else:
            name = to_cn(en_name)
        comps = ev.get("competitions", [{}])[0]
        status = comps.get("status", {}).get("type", {}).get("name", "")
        completed = comps.get("status", {}).get("type", {}).get("completed", False)
        
        date_str = ev.get("date", "")
        try:
            kickoff = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            kickoff = now_utc
        time_to = (kickoff - now_utc).total_seconds() / 3600
        
        competitors = comps.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        
        home_name = to_cn(home["team"]["displayName"]) if home else "?"
        away_name = to_cn(away["team"]["displayName"]) if away else "?"
        home_abbr = home["team"]["abbreviation"] if home else ""
        away_abbr = away["team"]["abbreviation"] if away else ""
        home_score = home.get("score", "0") if home else "0"
        away_score = away.get("score", "0") if away else "0"
        
        home_form = home.get("form", "") if home else ""
        away_form = away.get("form", "") if home else ""
        home_records = home.get("records", []) if home else []
        away_records = away.get("records", []) if away else []
        
        odds_raw = comps.get("odds") or []
        odds = next((o for o in odds_raw if o), {}) if odds_raw else {}
        
        details = odds.get("details", "")
        draw_ml = (odds.get("drawOdds") or {}).get("moneyLine", None)
        
        ps = odds.get("pointSpread") or {}
        spread_h = ps.get("home") or {}
        spread_a = ps.get("away") or {}
        spread_h_open = spread_h.get("open") or {}
        spread_h_close = spread_h.get("close") or {}
        spread_a_open = spread_a.get("open") or {}
        spread_a_close = spread_a.get("close") or {}
        
        tot = odds.get("total") or {}
        tot_o = tot.get("over") or {}
        tot_u = tot.get("under") or {}
        tot_o_close = tot_o.get("close") or {}
        tot_u_close = tot_u.get("close") or {}
        
        spread_h_line = spread_h_close.get("line", "")
        spread_h_odds = spread_h_close.get("odds", "")
        
        ml_team, ml_odds_str, home_ml_implied = parse_details(details)
        draw_implied = parse_american_odds(draw_ml)
        
        home_true, draw_true, away_true = remove_vig(home_ml_implied, draw_implied)
        
        spread_move = spread_movement_factor(spread_a_close)
        
        h_fs = form_to_score(home_form)
        a_fs = form_to_score(away_form)
        h_rs = record_to_score(home_records)
        a_rs = record_to_score(away_records)
        
        rec = {
            "name": name,
            "status": status,
            "completed": completed,
            "kickoff_utc": date_str,
            "time_to_kickoff_h": round(time_to, 1),
            "home": home_name,
            "away": away_name,
            "home_en": home["team"]["displayName"] if home else "",
            "away_en": away["team"]["displayName"] if away else "",
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "score": f"{home_score}-{away_score}" if status in ("STATUS_FULL_TIME", "STATUS_FINAL_PEN", "STATUS_FINAL_ET") else "",
            "home_form": home_form,
            "away_form": away_form,
            "home_form_score": round(h_fs, 3),
            "away_form_score": round(a_fs, 3),
            "home_record": home_records[0].get("summary","") if home_records else "",
            "away_record": away_records[0].get("summary","") if away_records else "",
            "home_record_score": round(h_rs, 3),
            "away_record_score": round(a_rs, 3),
            "ml_home_close": ml_odds_str,
            "draw_ml": draw_ml,
            "home_ml_implied": round(home_ml_implied, 4) if home_ml_implied else None,
            "draw_implied": round(draw_implied, 4) if draw_implied else None,
            "home_true_prob": round(home_true, 4) if home_true else None,
            "draw_true_prob": round(draw_true, 4) if draw_true else None,
            "away_true_prob": round(away_true, 4) if away_true else None,
            "spread_home_line": spread_h_line,
            "spread_home_close_odds": spread_h_odds,
            "spread_movement_score": round(spread_move, 3),
            "total_over_close": tot_o_close.get("odds",""),
            "total_under_close": tot_u_close.get("odds",""),
        }
        
        if status in ("STATUS_FULL_TIME", "STATUS_FINAL_PEN", "STATUS_FINAL_ET"):
            past.append(rec)
        elif status == "STATUS_SCHEDULED":
            future.append(rec)
        else:
            in_progress.append(rec)
    
    return past, future, in_progress


# ─── 主预测函数（Onside 4 信号 + Dixon-Coles） ──
def calculate_prediction(match, weights=None, calibration_offset=None,
                         fifa_rankings=None, host_country=None, use_dixon_coles=True):
    """
    Onside 4 信号 + Dixon-Coles 预测 → 方向 + 信心 + 比分预测 + 95% CI
    
    Args:
        match: 比赛数据字典
        weights: 权重字典（可选，默认 ONSIDE_WEIGHTS）
        calibration_offset: 校准偏移字典
        fifa_rankings: FIFA 排名字典
        host_country: 东道主国家
        use_dixon_coles: 是否使用 Dixon-Coles 模型
    """
    if weights is None:
        weights = ONSIDE_WEIGHTS
    
    hp = match.get("home_true_prob") or 0.5
    dp = match.get("draw_true_prob") or 0.25
    ap = match.get("away_true_prob") or 0.25
    hfs = match.get("home_form_score", 0.5)
    afs = match.get("away_form_score", 0.5)
    hrs = match.get("home_record_score", 0.5)
    ars = match.get("away_record_score", 0.5)
    sm = match.get("spread_movement_score", 0)
    
    # ── Onside 4 信号 ──
    home_en = match.get("home_en", match.get("home", ""))
    away_en = match.get("away_en", match.get("away", ""))
    
    if fifa_rankings is None:
        fifa_rankings = fetch_fifa_rankings()
    
    onside = compute_onside_signals(home_en, away_en, fifa_rankings, host_country)
    home_onside = onside["home"]["onside_score"]
    away_onside = onside["away"]["onside_score"]
    
    # ── 应用 calibration offset 修正隐含概率 ──
    calibration_note = None
    if calibration_offset:
        hc = calibration_offset.get("home_correction", 1.0)
        dc = calibration_offset.get("draw_correction", 1.0)
        ac = calibration_offset.get("away_correction", 1.0)
        
        # 修正：乘以 offset 后重新归一化
        hp_corrected = hp * hc
        dp_corrected = dp * dc
        ap_corrected = ap * ac
        total_corrected = hp_corrected + dp_corrected + ap_corrected
        if total_corrected > 0:
            hp = hp_corrected / total_corrected
            dp = dp_corrected / total_corrected
            ap = ap_corrected / total_corrected
        
        calibration_note = f"calibration applied (n={calibration_offset.get('sample_size','?')}, " \
                           f"home×{hc}/draw×{dc}/away×{ac})"
        log(calibration_note)
    
    sm_capped = max(-0.15, min(0.15, sm))
    
    # ── Onside 4 信号加权评分 ──
    # 市场信号 (20%): 盘口隐含概率
    market_home = hp
    market_draw = dp
    market_away = ap
    
    # 综合评分 = 市场信号×20% + Onside信号×80%
    home_strength = (
        market_home * weights["market_odds"]
        + home_onside * (1 - weights["market_odds"])
        + sm_capped * 0.5
    )
    away_strength = (
        market_away * weights["market_odds"]
        + away_onside * (1 - weights["market_odds"])
        + (-sm_capped) * 0.5
    )
    draw_strength = max(0, market_draw * weights["market_odds"] + 0.15)
    
    # 东道主额外加成
    if host_country and home_en == host_country:
        home_strength += HOST_ADVANTAGE_BOOST
    
    total = max(home_strength + draw_strength + away_strength, 0.05)
    home_prob = home_strength / total
    draw_prob_calc = draw_strength / total
    away_prob = away_strength / total
    
    # ── 方向判断 ──
    if home_prob > 0.45 and home_prob > away_prob * 1.3:
        direction = f"{match['home']} 胜"
        confidence_raw = (home_prob - 0.25) * 2
    elif away_prob > 0.45 and away_prob > home_prob * 1.3:
        direction = f"{match['away']} 胜"
        confidence_raw = (away_prob - 0.25) * 2
    elif draw_prob_calc > 0.40:
        direction = "平局"
        confidence_raw = (draw_prob_calc - 0.25) * 2
    else:
        if home_prob >= away_prob and home_prob >= draw_prob_calc:
            direction = f"{match['home']} 胜 (接近)"
            confidence_raw = (home_prob - 0.33) * 3
        elif away_prob >= home_prob and away_prob >= draw_prob_calc:
            direction = f"{match['away']} 胜 (接近)"
            confidence_raw = (away_prob - 0.33) * 3
        else:
            direction = "平局 (接近)"
            confidence_raw = (draw_prob_calc - 0.33) * 3
    
    confidence_raw = min(max(confidence_raw, 0.0), 1.0)
    if confidence_raw >= 0.90:
        stars = "⭐⭐⭐⭐⭐"
    elif confidence_raw >= 0.72:
        stars = "⭐⭐⭐⭐"
    elif confidence_raw >= 0.55:
        stars = "⭐⭐⭐"
    elif confidence_raw >= 0.35:
        stars = "⭐⭐"
    else:
        stars = "⭐"
    
    # ── 期望进球计算 ──
    LAMBDA_MULTIPLIER = 4.5
    raw_home = hp * 0.40 + hfs * 0.20 + hrs * 0.15 + sm * 0.25
    raw_away = ap * 0.40 + afs * 0.20 + ars * 0.15 + (-sm) * 0.25
    raw_draw = dp * 0.50
    lambda_home = max((raw_home + 0.5 * raw_draw) * LAMBDA_MULTIPLIER, 0.3)
    lambda_away = max((raw_away + 0.5 * raw_draw) * LAMBDA_MULTIPLIER, 0.3)
    
    # ── Dixon-Coles 或独立泊松比分预测 ──
    if use_dixon_coles:
        dc_result = dixon_coles_match_probs(lambda_home, lambda_away, rho=DC_RHO)
        top3 = [(s[0], s[1], round(s[2], 4)) for s in dc_result["score_probs"][:3]]
        predicted_score = f"{top3[0][0]}-{top3[0][1]}"
        
        # 从 DC 模型计算 BTTS 和 Over/2.5
        btts_prob = sum(s[2] for s in dc_result["score_probs"] if s[0] > 0 and s[1] > 0)
        over_25_prob = sum(s[2] for s in dc_result["score_probs"] if s[0] + s[1] > 2)
    else:
        probs = []
        for h in range(9):
            for a in range(9):
                p = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)
                if p >= 0.001:
                    probs.append((h, a, p))
        probs.sort(key=lambda x: -x[2])
        top3 = probs[:3]
        predicted_score = f"{top3[0][0]}-{top3[0][1]}"
        
        btts_prob = sum(p[2] for p in probs if p[0] > 0 and p[1] > 0)
        over_25_prob = sum(p[2] for p in probs if p[0] + p[1] > 2)
    
    # 95% 置信区间
    ci_home = poisson_confidence_interval(lambda_home)
    ci_away = poisson_confidence_interval(lambda_away)
    
    ou_total = match.get("total_over_close", "2.5")
    if over_25_prob > 0.5:
        ou = f"Over {ou_total}"
    else:
        ou = f"Under {ou_total}"
    
    return {
        "direction": direction,
        "stars": stars,
        "confidence_score": round(confidence_raw, 3),
        "predicted_score": predicted_score,
        "poisson_top3": [
            {"score": f"{h}-{a}", "prob": round(p, 4)} for h, a, p in top3
        ],
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "lambda_home_ci95": ci_home,
        "lambda_away_ci95": ci_away,
        "over_under": f"{ou} @ {match.get('total_over_close','')}" if match.get('total_over_close','') else f"{ou}",
        "btts": "Yes" if btts_prob > 0.5 else "No",
        "dixon_coles_used": use_dixon_coles,
        "dixon_coles_rho": DC_RHO if use_dixon_coles else None,
        "onside_signals": onside,
        "reasoning_factors": {
            "home_ml_true_prob": round(hp, 3),
            "draw_true_prob": round(dp, 3),
            "away_ml_true_prob": round(ap, 3),
            "home_form_score": round(hfs, 3),
            "away_form_score": round(afs, 3),
            "home_record_score": round(hrs, 3),
            "away_record_score": round(ars, 3),
            "spread_movement": round(sm, 3),
            "home_onside_score": round(home_onside, 3),
            "away_onside_score": round(away_onside, 3),
            "home_prob_weighted": round(home_prob, 3),
            "draw_prob_weighted": round(draw_prob_calc, 3),
            "away_prob_weighted": round(away_prob, 3),
        }
    }


# ─── 蒙特卡洛冠军模拟 ──────────────────────────
def simulate_match_dc(lambda_h, lambda_a, rho=DC_RHO):
    """
    用 Dixon-Coles 模型模拟单场比赛结果。
    
    Args:
        lambda_h: 主队期望进球
        lambda_a: 客队期望进球
        rho: 相关系数
    
    Returns:
        tuple: (home_goals, away_goals)
    """
    # 生成概率分布并采样
    probs = []
    for h in range(8):
        for a in range(8):
            p = dixon_coles_pmf(h, a, lambda_h, lambda_a, rho)
            probs.append((h, a, p))
    
    # 归一化
    total_p = sum(p[2] for p in probs)
    if total_p > 0:
        probs = [(h, a, p / total_p) for h, a, p in probs]
    
    # 采样
    r = random.random()
    cumulative = 0.0
    for h, a, p in probs:
        cumulative += p
        if r <= cumulative:
            return h, a
    
    # fallback
    return 0, 0


def monte_carlo_champion(fixtures, team_strengths, n_simulations=DEFAULT_N_SIMULATIONS,
                         rho=DC_RHO, tournament_type="world_cup"):
    """
    蒙特卡洛模拟整个锦标赛，预测冠军概率。
    
    Args:
        fixtures: 赛程列表，每项为 {"home": str, "away": str, "stage": str, "group": str}
        team_strengths: 各队实力评分 {team_name: {lambda_home: float, lambda_away: float, ...}}
        n_simulations: 模拟次数（默认 10000）
        rho: Dixon-Coles 相关系数
        tournament_type: "world_cup" 或 "league"
    
    Returns:
        dict: {
            "champion_probs": {team: prob},
            "round_reach_probs": {round_name: {team: prob}},
            "simulation_count": int,
            "model": "dixon_coles"
        }
    """
    log(f"Starting Monte Carlo simulation: {n_simulations} iterations, type={tournament_type}")
    
    champion_counts = {}
    round_reach_counts = {}
    
    # 收集所有队伍
    all_teams = set()
    for f in fixtures:
        all_teams.add(f.get("home", ""))
        all_teams.add(f.get("away", ""))
    
    for team in all_teams:
        champion_counts[team] = 0
        round_reach_counts[team] = {}
    
    for sim in range(n_simulations):
        if sim % 2000 == 0 and sim > 0:
            log(f"  Simulation {sim}/{n_simulations}...")
        
        if tournament_type == "world_cup":
            result = simulate_world_cup(fixtures, team_strengths, rho)
        else:
            result = simulate_league(fixtures, team_strengths, rho)
        
        # 记录冠军
        champion = result.get("champion")
        if champion:
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
        
        # 记录各轮次到达
        for team, rounds in result.get("team_rounds", {}).items():
            for round_name in rounds:
                if round_name not in round_reach_counts[team]:
                    round_reach_counts[team][round_name] = 0
                round_reach_counts[team][round_name] += 1
    
    # 计算概率
    champion_probs = {team: round(count / n_simulations, 4) 
                      for team, count in champion_counts.items() if count > 0}
    champion_probs = dict(sorted(champion_probs.items(), key=lambda x: -x[1]))
    
    round_reach_probs = {}
    for team, rounds in round_reach_counts.items():
        for round_name, count in rounds.items():
            if round_name not in round_reach_probs:
                round_reach_probs[round_name] = {}
            round_reach_probs[round_name][team] = round(count / n_simulations, 4)
    
    # 排序
    for round_name in round_reach_probs:
        round_reach_probs[round_name] = dict(
            sorted(round_reach_probs[round_name].items(), key=lambda x: -x[1])
        )
    
    return {
        "champion_probs": champion_probs,
        "round_reach_probs": round_reach_probs,
        "simulation_count": n_simulations,
        "model": "dixon_coles",
        "rho": rho,
    }


def simulate_world_cup(fixtures, team_strengths, rho):
    """
    模拟一次世界杯锦标赛。
    
    简化模型：
    - 小组赛：每组4队，前2名晋级
    - 淘汰赛：单场定胜负（含加时/点球简化）
    """
    # 按阶段分组
    groups = {}
    knockout = []
    
    for f in fixtures:
        stage = f.get("stage", "group")
        if stage == "group":
            group_name = f.get("group", "A")
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(f)
        else:
            knockout.append(f)
    
    # 模拟小组赛
    group_standings = {}
    team_rounds = {}
    
    for group_name, group_fixtures in groups.items():
        # 初始化小组积分
        teams_in_group = set()
        for f in group_fixtures:
            teams_in_group.add(f["home"])
            teams_in_group.add(f["away"])
        
        standings = {team: {"points": 0, "gf": 0, "ga": 0, "gd": 0} for team in teams_in_group}
        
        for f in group_fixtures:
            home = f["home"]
            away = f["away"]
            
            # 获取期望进球
            lh = team_strengths.get(home, {}).get("lambda_home", 1.5)
            la = team_strengths.get(away, {}).get("lambda_away", 1.2)
            
            # 模拟
            hg, ag = simulate_match_dc(lh, la, rho)
            
            standings[home]["gf"] += hg
            standings[home]["ga"] += ag
            standings[home]["gd"] += hg - ag
            standings[away]["gf"] += ag
            standings[away]["ga"] += hg
            standings[away]["gd"] += ag - hg
            
            if hg > ag:
                standings[home]["points"] += 3
            elif hg == ag:
                standings[home]["points"] += 1
                standings[away]["points"] += 1
            else:
                standings[away]["points"] += 3
        
        # 排序（积分→净胜球→进球）
        sorted_teams = sorted(standings.keys(), 
                              key=lambda t: (standings[t]["points"], standings[t]["gd"], standings[t]["gf"]),
                              reverse=True)
        
        group_standings[group_name] = sorted_teams
        
        # 记录所有队伍到达小组赛
        for team in teams_in_group:
            if team not in team_rounds:
                team_rounds[team] = []
            team_rounds[team].append("group_stage")
        
        # 前2名晋级
        advanced = sorted_teams[:2]
        for team in advanced:
            if team not in team_rounds:
                team_rounds[team] = []
            team_rounds[team].append("round_of_16")
    
    # 模拟淘汰赛（简化：按赛程顺序）
    # 实际世界杯有固定对阵，这里简化处理
    current_round = "round_of_16"
    remaining_teams = []
    
    # 收集所有晋级队伍
    for group_name in sorted(group_standings.keys()):
        advanced = group_standings[group_name][:2]
        remaining_teams.extend(advanced)
    
    # 如果没有明确的淘汰赛赛程，自动生成对阵
    if not knockout and len(remaining_teams) >= 2:
        # 生成淘汰赛对阵
        while len(remaining_teams) > 1:
            next_round_teams = []
            for i in range(0, len(remaining_teams), 2):
                if i + 1 < len(remaining_teams):
                    t1 = remaining_teams[i]
                    t2 = remaining_teams[i + 1]
                    
                    lh = team_strengths.get(t1, {}).get("lambda_home", 1.5)
                    la = team_strengths.get(t2, {}).get("lambda_away", 1.2)
                    
                    hg, ag = simulate_match_dc(lh, la, rho)
                    
                    # 淘汰赛平局 → 简化：50% 概率晋级
                    if hg == ag:
                        winner = t1 if random.random() < 0.5 else t2
                    elif hg > ag:
                        winner = t1
                    else:
                        winner = t2
                    
                    next_round_teams.append(winner)
                    
                    # 记录轮次
                    round_name = f"r{len(remaining_teams)}"
                    if winner not in team_rounds:
                        team_rounds[winner] = []
                    team_rounds[winner].append(round_name)
                else:
                    # 轮空
                    next_round_teams.append(remaining_teams[i])
            
            remaining_teams = next_round_teams
            if len(remaining_teams) > 1:
                current_round = f"r{len(remaining_teams)}"
    
    champion = remaining_teams[0] if remaining_teams else None
    
    return {
        "champion": champion,
        "team_rounds": team_rounds,
    }


def simulate_league(fixtures, team_strengths, rho):
    """
    模拟一次联赛赛季。
    
    联赛模式：双循环，积分制，最高分夺冠。
    """
    # 初始化积分
    standings = {}
    
    for f in fixtures:
        home = f["home"]
        away = f["away"]
        
        if home not in standings:
            standings[home] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        if away not in standings:
            standings[away] = {"points": 0, "gf": 0, "ga": 0, "gd": 0}
        
        # 模拟
        lh = team_strengths.get(home, {}).get("lambda_home", 1.5)
        la = team_strengths.get(away, {}).get("lambda_away", 1.2)
        
        hg, ag = simulate_match_dc(lh, la, rho)
        
        standings[home]["gf"] += hg
        standings[home]["ga"] += ag
        standings[home]["gd"] += hg - ag
        standings[away]["gf"] += ag
        standings[away]["ga"] += hg
        standings[away]["gd"] += ag - hg
        
        if hg > ag:
            standings[home]["points"] += 3
        elif hg == ag:
            standings[home]["points"] += 1
            standings[away]["points"] += 1
        else:
            standings[away]["points"] += 3
    
    # 排序
    sorted_teams = sorted(standings.keys(),
                          key=lambda t: (standings[t]["points"], standings[t]["gd"], standings[t]["gf"]),
                          reverse=True)
    
    champion = sorted_teams[0] if sorted_teams else None
    
    team_rounds = {}
    for team in standings:
        team_rounds[team] = ["season"]
    
    return {
        "champion": champion,
        "team_rounds": team_rounds,
    }


# ─── 校准 ──────────────────────────────────────
def build_calibration(past_matches, future_matches):
    """从结束比赛计算校准参数"""
    if not past_matches:
        return {"note": "no past matches to calibrate from"}
    
    home_wins = sum(1 for m in past_matches if m["score"] and m["score"].split("-")[0].isdigit() and m["score"].split("-")[1].isdigit() and int(m["score"].split("-")[0]) > int(m["score"].split("-")[1]))
    draws = sum(1 for m in past_matches if m["score"] and m["score"].split("-")[0] == m["score"].split("-")[1])
    away_wins = sum(1 for m in past_matches if m["score"] and m["score"].split("-")[0].isdigit() and m["score"].split("-")[1].isdigit() and int(m["score"].split("-")[0]) < int(m["score"].split("-")[1]))
    total = home_wins + draws + away_wins
    
    favorite_wins = 0
    total_odds_based = 0
    for m in past_matches:
        hp = m.get("home_true_prob")
        if hp and hp > 0.5:
            total_odds_based += 1
            if m["score"]:
                try:
                    hs, aws = m["score"].split("-")
                    if int(hs) > int(aws):
                        favorite_wins += 1
                except: pass
    
    return {
        "total_matches": total,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_rate": round(home_wins/total, 3) if total else 0,
        "draw_rate": round(draws/total, 3) if total else 0,
        "away_win_rate": round(away_wins/total, 3) if total else 0,
        "favored_by_odds": total_odds_based,
        "favored_won": favorite_wins,
        "odds_accuracy": round(favorite_wins/total_odds_based, 3) if total_odds_based else 0,
    }


# ─── 主函数 ────────────────────────────────────
def main():
    # ─── 清理模式 ──────────────────────────────────
    if "--cleanup" in sys.argv:
        cleanup_old_files(days=7)
        return
    
    # ─── 解析参数 ──────────────────────────────────
    now_utc = datetime.now(timezone.utc)
    
    # 联赛选择
    league_key = "wc"
    for arg in sys.argv:
        if arg.startswith("--league="):
            league_key = arg.split("=")[1].lower()
    
    # 数据源选择
    data_source = None
    for arg in sys.argv:
        if arg.startswith("--data-source="):
            data_source = arg.split("=")[1].lower()
    
    # 蒙特卡洛
    run_monte_carlo = "--monte-carlo" in sys.argv
    n_simulations = DEFAULT_N_SIMULATIONS
    for arg in sys.argv:
        if arg.startswith("--n-simulations="):
            n_simulations = int(arg.split("=")[1])
    
    # Dixon-Coles
    use_dc = "--no-dc" not in sys.argv  # 默认启用
    
    # 日期范围
    d1 = (now_utc - timedelta(days=1)).strftime("%Y%m%d")
    d2 = (now_utc + timedelta(days=1)).strftime("%Y%m%d")
    dates_str = f"{d1}-{d2}"
    
    skip_fetch = "--no-fetch" in sys.argv
    
    # 联赛配置
    league_config = LEAGUE_CONFIG.get(league_key, LEAGUE_CONFIG["wc"])
    if data_source is None:
        data_source = league_config["data_source"]
    
    host_country = league_config.get("host_country")
    tournament_type = league_config.get("tournament_type", "world_cup")
    
    log(f"League: {league_key} ({league_config['name']}), source: {data_source}, type: {tournament_type}")
    
    # ─── 获取数据 ──────────────────────────────────
    if skip_fetch:
        with open("/tmp/espn_wc.json") as f:
            data = json.load(f)
        events = data.get("events", [])
    else:
        events = fetch_events(dates_str, league_key, data_source)
    
    log(f"Got {len(events)} events")
    
    past, future, in_prog = parse_events(events, now_utc)
    log(f"Past: {len(past)}, Future: {len(future)}, In progress: {len(in_prog)}")
    
    # ─── FIFA 排名 ─────────────────────────────────
    fifa_rankings = fetch_fifa_rankings()
    log(f"FIFA rankings loaded: {len(fifa_rankings)} teams")
    
    # ─── 赛事空窗期检测 ────────────────────────────
    if not future and not past:
        log("No matches found in window — outputting empty result")
        output = {
            "generated_at": now_utc.isoformat(),
            "data_window": dates_str,
            "status": "no_matches",
            "league": league_key,
            "tournament_type": tournament_type,
            "message": f"未来 24h 内无比赛（{dates_str}）",
            "calibration": {"note": "no data"},
            "past_matches": [],
            "predictions": [],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return
    
    if not future:
        log("No future matches to predict — outputting calibration only")
        calibration = build_calibration(past, future)
        output = {
            "generated_at": now_utc.isoformat(),
            "data_window": dates_str,
            "status": "no_future_matches",
            "league": league_key,
            "tournament_type": tournament_type,
            "message": f"未来 24h 内无待预测比赛（{dates_str}）",
            "calibration": calibration,
            "past_matches": past,
            "predictions": [],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
        # 仍然保存快照
        ts = now_utc.strftime("%Y-%m-%d_%H")
        pred_file = PREDICTIONS_DIR / f"prediction_{ts}.json"
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(pred_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log(f"Saved (no predictions): {pred_file}")
        return
    
    # ─── 正常预测流程 ──────────────────────────────
    calibration = build_calibration(past, future)
    log(f"Calibration: {json.dumps(calibration)}")
    
    # ── 累积校准：从历史 past_matches 计算修正因子 ──
    historical_past = load_historical_past_matches(days=30)
    calibration_offset = compute_calibration_offset(historical_past)
    if calibration_offset:
        log(f"Calibration offset: {json.dumps(calibration_offset)}")
    else:
        log("Calibration offset: insufficient historical data (<5 matches)")
    
    # ── 预测 ──────────────────────────────────────
    predictions = []
    for m in sorted(future, key=lambda x: x.get("time_to_kickoff_h", 0))[:5]:
        if -24 <= m.get("time_to_kickoff_h", 24) <= 24:
            pred = calculate_prediction(
                m, 
                calibration_offset=calibration_offset,
                fifa_rankings=fifa_rankings,
                host_country=host_country,
                use_dixon_coles=use_dc,
            )
            predictions.append({
                "match": m["name"],
                "home": m["home"],
                "away": m["away"],
                "kickoff_utc": m["kickoff_utc"],
                "time_to_kickoff_h": m["time_to_kickoff_h"],
                **pred
            })
    
    # ── 蒙特卡洛模拟 ──────────────────────────────
    monte_carlo_result = None
    if run_monte_carlo and len(future) >= 2:
        log(f"Running Monte Carlo simulation ({n_simulations} iterations)...")
        
        # 构建队伍实力字典
        team_strengths = {}
        for m in future:
            home_en = m.get("home_en", m.get("home", ""))
            away_en = m.get("away_en", m.get("away", ""))
            
            if home_en not in team_strengths:
                team_strengths[home_en] = {"lambda_home": 1.5, "lambda_away": 1.2}
            if away_en not in team_strengths:
                team_strengths[away_en] = {"lambda_home": 1.5, "lambda_away": 1.2}
        
        # 构建赛程
        mc_fixtures = []
        for m in future:
            mc_fixtures.append({
                "home": m.get("home_en", m.get("home", "")),
                "away": m.get("away_en", m.get("away", "")),
                "stage": "group",
                "group": "A",
            })
        
        monte_carlo_result = monte_carlo_champion(
            mc_fixtures, team_strengths, 
            n_simulations=n_simulations,
            rho=DC_RHO,
            tournament_type=tournament_type,
        )
        
        log(f"Monte Carlo complete. Top champion: {list(monte_carlo_result['champion_probs'].items())[:3]}")
    
    # ── 输出 ──────────────────────────────────────
    output = {
        "generated_at": now_utc.isoformat(),
        "data_window": dates_str,
        "status": "ok",
        "league": league_key,
        "tournament_type": tournament_type,
        "data_source": data_source,
        "dixon_coles_enabled": use_dc,
        "dixon_coles_rho": DC_RHO if use_dc else None,
        "calibration": calibration,
        "calibration_offset": calibration_offset,
        "past_matches": past,
        "predictions": predictions,
    }
    
    if monte_carlo_result:
        output["monte_carlo"] = monte_carlo_result
    
    print(json.dumps(output, indent=2, ensure_ascii=False))
    
    ts = now_utc.strftime("%Y-%m-%d_%H")
    pred_file = PREDICTIONS_DIR / f"prediction_{ts}.json"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pred_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Saved: {pred_file}")
    
    with open("/tmp/pred_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    
    # ─── 输出摘要到 stderr ─────────────────────────
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"📊 窗口校准: {calibration.get('total_matches',0)}场已结束 | 主胜 {calibration.get('home_win_rate',0)*100:.0f}% 平 {calibration.get('draw_rate',0)*100:.0f}% 客胜 {calibration.get('away_win_rate',0)*100:.0f}%", file=sys.stderr)
    print(f"   投注热门正确率: {calibration.get('odds_accuracy',0)*100:.0f}% ({calibration.get('favored_won',0)}/{calibration.get('favored_by_odds',0)})", file=sys.stderr)
    if calibration_offset:
        print(f"🔧 累积校准(n={calibration_offset['sample_size']}): 主×{calibration_offset['home_correction']} 平×{calibration_offset['draw_correction']} 客×{calibration_offset['away_correction']}", file=sys.stderr)
        print(f"   实际分布: 主 {calibration_offset['actual_home_rate']} | 平 {calibration_offset['actual_draw_rate']} | 客 {calibration_offset['actual_away_rate']}", file=sys.stderr)
    else:
        print(f"🔧 累积校准: 数据不足（<5场），跳过校准", file=sys.stderr)
    print(f"🔥 待预测: {len(predictions)} 场", file=sys.stderr)
    for p in predictions:
        poisson_str = " / ".join(f"{t['score']}({t['prob']:.0%})" for t in p.get('poisson_top3', [])[:3])
        ci_home = p.get('lambda_home_ci95', (0,0))
        ci_away = p.get('lambda_away_ci95', (0,0))
        cal = ' 📐cal' if calibration_offset else ''
        dc = ' 🎯DC' if p.get('dixon_coles_used') else ''
        print(f"  {p['match']} | {p['direction']} {p['stars']}{cal}{dc} | {p['predicted_score']} | λ={p.get('lambda_home',0)}[{ci_home[0]}-{ci_home[1]}]/{p.get('lambda_away',0)}[{ci_away[0]}-{ci_away[1]}] | {poisson_str}", file=sys.stderr)
    
    if monte_carlo_result:
        print(f"\n🏆 蒙特卡洛冠军预测 (n={n_simulations}):", file=sys.stderr)
        for team, prob in list(monte_carlo_result["champion_probs"].items())[:5]:
            print(f"  {team}: {prob:.1%}", file=sys.stderr)
    
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()

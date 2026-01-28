'''
Jack Miller
January 2026
'''


''' Imports '''

import pandas as pd
import polars as pl
import numpy as np

import plotly.express as px

import nflreadpy as nfl
import nfl_data_py as nfldy


''' Parameters / Constants '''

START_YEAR = 2018       # first year of participation data
END_YEAR = 2024
SEASONS = [i for i in range(START_YEAR, END_YEAR + 1)]



''' Helper Functions '''

def clean_personnel(personnel_str: str) -> str:
    if not personnel_str:
        return ''
    
    personnel_list = personnel_str.split(', ')
    personnel_str_positions = ''
    for i in personnel_list:

        num = int(i.split(' ')[0])
        pos = i.split(' ')[1]
        pos_str = f'{pos};' * num

        personnel_str_positions += pos_str

    return personnel_str_positions

def offensive_personnel(personnel_str: str) -> str:
    if type(personnel_str) != str:
        return ''
    
    spts = personnel_str.count('K;') + personnel_str.count('P;') + personnel_str.count('LS;') + personnel_str.count('FS;') + personnel_str.count('CB;')
    if spts > 0:
        return 'ST'
    
    wrs = personnel_str.count('WR')
    rbs = personnel_str.count('RB')
    tes = personnel_str.count('TE')

    personnel = f'{rbs}{tes}'

    centers = personnel_str.count('C;')
    guards = personnel_str.count('G;')
    tackles = personnel_str.count('T;')
    ol = personnel_str.count('OL;')

    if (centers + guards + tackles) > 5 or ol > 5:
        asts = ''
        if (centers + guards + tackles) > 5:
            asts = '*' * ((centers + guards + tackles) - 5)
        else:
            asts = '*' * (ol - 5)    
                 
        personnel += asts

    return personnel

def defensive_personnel(personnel_str: str) -> str:
    if type(personnel_str) != str:
        return ''
    
    spts = personnel_str.count('K;') + personnel_str.count('P;') + personnel_str.count('LS;') + personnel_str.count('WR;') + personnel_str.count('RB;') + personnel_str.count('TE;')
    if spts > 0:
        return 'ST'
    
    # DL
    dls = personnel_str.count('DL;')
    des = personnel_str.count('DE;')
    dts = personnel_str.count('DT;')
    nts = personnel_str.count('NT;')
    total_dls = dls + des + dts + nts

    # LBs
    lbs = personnel_str.count('LB;')
    # mlbs = personnel_str.count('MLB;')
    # ilbs = personnel_str.count('ILB;')
    # olbs = personnel_str.count('OLB;')
    total_lbs = lbs #+ mlbs + ilbs + olbs

    # DBs
    dbs = personnel_str.count('DB;')
    cbs = personnel_str.count('CB;')
    sss = personnel_str.count('SS;')
    fss = personnel_str.count('FS;')
    total_dbs = dbs + cbs + sss + fss

    d_type = ''
    if total_dbs == 4: d_type = 'Base'
    elif total_dbs == 5: d_type = 'Nickel'
    elif total_dbs == 6: d_type = 'Dime'
    elif total_dbs == 7: d_type = 'Quarters'
    else: d_type = 'Other'

    return f'{d_type} {total_dls}-{total_lbs}'



''' PBP Data '''

## Load ##
pbp = nfl.load_pbp(seasons=SEASONS)

## Add columns ##
pbp = pbp.with_columns(
    MasterPlayID=pl.concat_str([pl.col('game_id'), pl.col('play_id').cast(pl.Int32).cast(pl.String)], separator='_'),
    DriveID=pl.concat_str([pl.col('game_id'), pl.col('drive').cast(pl.Int8).cast(pl.String)], separator='_'),
)
pbp = pbp.with_columns(
    NeutralDown=pl.when((pl.col('down') == 1) & (pl.col('ydstogo') <= 10)).then(1).when((pl.col('down') == 2) & (pl.col('ydstogo') <= 6)).then(1).when((pl.col('down') == 3) & (pl.col('ydstogo') <= 3)).then(1).otherwise(0),
    PassDepth=pl.when(pl.col('air_yards') <= 0).then(pl.lit('Behind LOS')).when(pl.col('air_yards') < 10).then(pl.lit('Short')).when(pl.col('air_yards') < 20).then(pl.lit('Medium')).when(pl.col('air_yards') >= 20).then(pl.lit('Long')),
    AirYardsToSticks=pl.col('air_yards') - pl.col('ydstogo'),

    # Trouble with run_gap and the middle rush https://thespade.substack.com/p/run-gap-charts-version-15
    RunLocation=pl.when(pl.col('run_gap') == 'end').then(pl.lit('Outside')).when((pl.col('run_gap') == 'guard') | (pl.col('run_gap') == 'tackle')).then(pl.lit('Inside'))
)

## Filters ##

# Filter to relevant plays (see nflfastr beginner's guide)
pbp = pbp.filter(
    (pl.col('pass') == 1) | (pl.col('rush') == 1),
    (pl.col('season_type') == 'REG'),
    (pl.col('epa').is_not_nan()),
    (pl.col('posteam').is_not_null()),
    (pl.col('posteam') != ''),
)

# Filter to normal game state
pbp = pbp.filter(
    # (pl.col('qtr') <= 3),
    # (pl.col('half_seconds_remaining') > 120),
    # (pl.col('score_differential') <= 14),
    (pl.col('special_teams_play') == 0),
    (pl.col('play_type_nfl') != 'PAT2'),
    (pl.col('play_type_nfl') != 'UNSPECIFIED'),     # Unspecified seems to be mostly punt / FG formation plays where something weird happened (fake, fumble, botched snap, etc)
)



''' Participation Data '''
    
## Get data ##
participation = nfl.load_participation(seasons=SEASONS)

## Add columns
participation = participation.with_columns(
    MasterPlayID=pl.concat_str([pl.col('nflverse_game_id'), pl.col('play_id').cast(pl.Int32).cast(pl.String)], separator='_'),
    season=pl.col('nflverse_game_id').str.split('_').list.get(0).cast(int),

    # Defense stuff
    LightBox=pl.when(pl.col('defenders_in_box') <= 6).then(1).otherwise(0),
    HeavyBox=pl.when(pl.col('defenders_in_box') >= 8).then(1).otherwise(0),
    ZoneCoverage=pl.when(pl.col('defense_man_zone_type') == 'ZONE_COVERAGE').then(1).otherwise(0),
    ManCoverage=pl.when(pl.col('defense_man_zone_type') == 'MAN_COVERAGE').then(1).otherwise(0),
    OffenseFormation=pl.when(pl.col('offense_formation').is_in(['SINGLEBACK', 'I_FORM', 'UNDER CENTER', 'JUMBO'])).then(pl.lit('Under Center')).when(pl.col('offense_formation').is_in(['SHOTGUN', 'EMPTY', 'WILDCAT', 'PISTOL'])).then(pl.lit('Shotgun'))
)

# Personnel
participation = participation.with_columns(
    OffensePositionsStr=pl.col('offense_personnel').map_elements(clean_personnel, return_dtype=str),
    DefensePositionsStr=pl.col('defense_personnel').map_elements(clean_personnel, return_dtype=str),
)
participation = participation.with_columns(
    OffensePersonnelGroup=pl.col('OffensePositionsStr').map_elements(offensive_personnel, return_dtype=str),
    DefensePersonnelGroup=pl.col('DefensePositionsStr').map_elements(defensive_personnel, return_dtype=str),
)
participation = participation.with_columns(
    OffenseMultRBs=pl.when(pl.col('OffensePersonnelGroup').str.slice(0, 1).is_in(['2', '3', '4'])).then(1).otherwise(0),
    OffenseZeroRBs=pl.when(pl.col('OffensePersonnelGroup').str.slice(0, 1) == '0').then(1).otherwise(0),
    OffenseMultTEs=pl.when(pl.col('OffensePersonnelGroup').str.slice(1, 1).is_in(['2', '3', '4'])).then(1).otherwise(0),
    OffenseZeroTEs=pl.when(pl.col('OffensePersonnelGroup').str.slice(1, 1) == '0').then(1).otherwise(0),
    OffenseExtraOL=pl.when(pl.col('OffensePersonnelGroup').str.tail(1) == '*').then(1).otherwise(0),
    DefensePersonnelType=pl.col('DefensePersonnelGroup').str.split(' ').list.get(0)
)


''' Combine and Create Pandas DF '''

pbp = pbp.join(participation[['MasterPlayID', 'OffenseFormation', 'OffensePersonnelGroup', 'OffenseMultRBs', 'OffenseZeroRBs', 'OffenseMultTEs', 'OffenseZeroTEs', 'OffenseExtraOL', 'time_to_throw', 'DefensePersonnelGroup', 'DefensePersonnelType', 'LightBox', 'HeavyBox', 'number_of_pass_rushers', 'ZoneCoverage', 'ManCoverage', 'defense_coverage_type']], on='MasterPlayID', how='left')

# Create dataframe
pbp_df = pd.DataFrame(columns=pbp.columns, data=pbp)


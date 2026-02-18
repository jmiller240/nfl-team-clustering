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

pl.Config.set_tbl_width_chars(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)



''' Parameters / Constants '''

START_YEAR = 2016       # first year of participation data
END_YEAR = 2024
SEASONS = [i for i in range(START_YEAR, END_YEAR + 1)]



''' Helpers '''

## Personnel Helper Functions ## 

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


''' Main Functions '''

def load_pbp_participation_data() -> pd.DataFrame:

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
        (pl.col('qtr') <= 3),
        (pl.col('half_seconds_remaining') > 120),
        (pl.col('score_differential') <= 14),
        (pl.col('special_teams_play') == 0),
        (pl.col('play_type_nfl') != 'PAT2'),
        (pl.col('play_type_nfl') != 'UNSPECIFIED'),     # Unspecified seems to be mostly punt / FG formation plays where something weird happened (fake, fumble, botched snap, etc)
    )

    # print(pbp.shape)
    # print(pbp['MasterPlayID'].n_unique())
    # print(pbp.head())


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
        DefenseCoverage=pl.when(pl.col('defense_coverage_type').is_in(['COVER_1', 'COVER_2', 'COVER_3', 'COVER_4', 'COVER_6'])).then(pl.col('defense_coverage_type')).otherwise(pl.lit('Other')),
        OffenseFormation=pl.when(pl.col('offense_formation').is_in(['SINGLEBACK', 'I_FORM', 'UNDER CENTER', 'JUMBO'])).then(pl.lit('Under Center')).when(pl.col('offense_formation').is_in(['SHOTGUN', 'EMPTY', 'WILDCAT', 'PISTOL'])).then(pl.lit('Shotgun'))
    )

    # Personnel
    participation = participation.with_columns(
        OffensePositionsStr=pl.col('offense_personnel').map_elements(clean_personnel, return_dtype=str),
        DefensePositionsStr=pl.col('defense_personnel').map_elements(clean_personnel, return_dtype=str),
    )
    participation = participation.with_columns(
        OffensePersonnel=pl.col('OffensePositionsStr').map_elements(offensive_personnel, return_dtype=str),
        DefensePersonnel=pl.col('DefensePositionsStr').map_elements(defensive_personnel, return_dtype=str),
    )
    participation = participation.with_columns(
        OffensePersonnelGroup=pl.when(pl.col('OffensePersonnel').str.slice(0, 2).is_in(['11', '12', '13', '21', '22'])).then(pl.col('OffensePersonnel').str.slice(0, 2)).otherwise(pl.lit('Other')),
        OffenseMultRBs=pl.when(pl.col('OffensePersonnel').str.slice(0, 1).is_in(['2', '3', '4'])).then(1).otherwise(0),
        OffenseZeroRBs=pl.when(pl.col('OffensePersonnel').str.slice(0, 1) == '0').then(1).otherwise(0),
        OffenseMultTEs=pl.when(pl.col('OffensePersonnel').str.slice(1, 1).is_in(['2', '3', '4'])).then(1).otherwise(0),
        OffenseZeroTEs=pl.when(pl.col('OffensePersonnel').str.slice(1, 1) == '0').then(1).otherwise(0),
        OffenseExtraOL=pl.when(pl.col('OffensePersonnel').str.tail(1) == '*').then(1).otherwise(0),
        DefensePersonnelType=pl.col('DefensePersonnel').str.split(' ').list.get(0)
    )
    participation = participation.with_columns(
        OffenseHeavyPersonnel=pl.when((pl.col('OffenseMultRBs') == 1) | (pl.col('OffenseMultTEs') == 1)).then(1).otherwise(0),
    )

    # print(participation.shape)
    # print(participation['MasterPlayID'].n_unique())
    # print(participation.filter(pl.col('season') == 2024, pl.col('route') != '').head(100))


    ''' Combine '''

    participation_cols = ['MasterPlayID', 'OffenseFormation', 'OffensePersonnel','OffensePersonnelGroup', 'OffenseMultRBs', 'OffenseZeroRBs', 'OffenseMultTEs', 'OffenseZeroTEs', 
                          'OffenseExtraOL', 'OffenseHeavyPersonnel', 'time_to_throw', 'DefensePersonnel', 'DefensePersonnelType', 'LightBox', 'HeavyBox', 'number_of_pass_rushers', 
                          'ZoneCoverage', 'ManCoverage', 'defense_coverage_type', 'DefenseCoverage']
    pbp = pbp.join(participation[participation_cols], on='MasterPlayID', how='left')

    # Create dataframe
    pbp_df = pd.DataFrame(columns=pbp.columns, data=pbp)

    # print(pbp_df.shape)
    # print(pbp_df.head().to_string())

    return pbp_df


def load_stats_team_tendencies_offense():
    ''' Prep Offensive Inputs '''

    ## Get data ##
    pbp_df = load_pbp_participation_data()
 
    ## Base Stats ##
    offense_team_tendencies = pbp_df.groupby(['posteam', 'season']).aggregate(
        # General
        Games=('game_id', 'nunique'),
        Drives=('DriveID', 'nunique'),
        Plays=('posteam', 'size'),
        Neutral_Down_Plays=('posteam', lambda x: x[pbp_df['NeutralDown'] == 1].shape[0]),

        # Play Types
        Pass_Plays=('pass', 'sum'),
        Neutral_Down_Pass=('pass', lambda x: x[pbp_df['NeutralDown'] == 1].sum()),
        Pass_Attempts=('pass_attempt', 'sum'),
        
        QBScrambles=('qb_scramble', 'sum'),

        # Passing
        IAY=('air_yards', 'sum'),
        IAY_ToSticks=('AirYardsToSticks', 'sum'),
        TotalTimeToThrow=('time_to_throw', 'sum'),
        Pass_BehindLOS=('pass_attempt', lambda x: x[pbp_df['PassDepth'] == 'Behind LOS'].sum()),
        Pass_Deep=('pass_attempt', lambda x: x[pbp_df['PassDepth'] == 'Long'].sum()),
        Sacks=('sack', 'sum'),

        # Rushing
        Rush_Plays=('rush', 'sum'),
        Rush_Attempts=('rush_attempt', 'sum'),
        Rush_Inside=('rush', lambda x: x[pbp_df['RunLocation'] == 'Inside'].sum()),
        Rush_Outside=('rush', lambda x: x[pbp_df['RunLocation'] == 'Outside'].sum()),

        # Personnel
        Plays_11_Personnel=('posteam', lambda x: x[pbp_df['OffensePersonnel'] == '11'].shape[0]),
        Plays_Heavy_Personnel=('OffenseHeavyPersonnel', 'sum'),
        Plays_Mult_RBs=('OffenseMultRBs', 'sum'),
        Plays_Zero_RBs=('OffenseZeroRBs', 'sum'),
        Plays_Mult_TEs=('OffenseMultTEs', 'sum'),
        Plays_Zero_TEs=('OffenseZeroTEs', 'sum'),
        Plays_Extra_OL=('OffenseExtraOL', 'sum')
    )

    # Overall numbers
    offense_team_tendencies['Plays / Game'] = offense_team_tendencies['Plays'] / offense_team_tendencies['Games']
    offense_team_tendencies['Drives / Game'] = offense_team_tendencies['Drives'] / offense_team_tendencies['Games']

    # Play Types
    offense_team_tendencies['% Pass'] = offense_team_tendencies['Pass_Plays'] / offense_team_tendencies['Plays']
    offense_team_tendencies['% Pass Neutral Downs'] = offense_team_tendencies['Neutral_Down_Pass'] / offense_team_tendencies['Neutral_Down_Plays']

    offense_team_tendencies['Scrambles / Game'] = offense_team_tendencies['QBScrambles'] / offense_team_tendencies['Games']

    # Passing numbers
    offense_team_tendencies['ADOT'] = offense_team_tendencies['IAY'] / (offense_team_tendencies['Pass_Attempts'] - offense_team_tendencies['Sacks'])
    offense_team_tendencies['ADOT to Sticks'] = offense_team_tendencies['IAY_ToSticks'] / (offense_team_tendencies['Pass_Attempts'] - offense_team_tendencies['Sacks'])
    offense_team_tendencies['Avg Time to Throw'] = offense_team_tendencies['TotalTimeToThrow'] / (offense_team_tendencies['Pass_Attempts'] - offense_team_tendencies['Sacks'])

    offense_team_tendencies['% Passes Behind LOS'] = offense_team_tendencies['Pass_BehindLOS'] / (offense_team_tendencies['Pass_Attempts'] - offense_team_tendencies['Sacks'])
    offense_team_tendencies['% Passes Deep'] = offense_team_tendencies['Pass_Deep'] / (offense_team_tendencies['Pass_Attempts'] - offense_team_tendencies['Sacks'])

    # Rushing numbers
    offense_team_tendencies['% Rush Inside'] = offense_team_tendencies['Rush_Inside'] / offense_team_tendencies['Rush_Plays']
    offense_team_tendencies['% Rush Outside'] = offense_team_tendencies['Rush_Outside'] / offense_team_tendencies['Rush_Plays']

    # Personnel
    for col in ['Plays_11_Personnel', 'Plays_Heavy_Personnel', 'Plays_Mult_RBs', 'Plays_Zero_RBs', 'Plays_Mult_TEs', 'Plays_Zero_TEs', 'Plays_Extra_OL']:
        cat = col.replace('Plays_', '').replace('_', ' ')
        col_name = f'% Plays {cat}'
        offense_team_tendencies[col_name] = offense_team_tendencies[col] / offense_team_tendencies['Plays']

    # Formations
    offense_formations = pbp_df.groupby(['posteam', 'season', 'OffenseFormation']).aggregate(
        Plays=('posteam', 'size'),
        Neutral_Down_Plays=('posteam', lambda x: x[pbp_df['NeutralDown'] == 1].shape[0]),

        Pass_Plays=('pass', 'sum'),
        Rush_Plays=('rush', 'sum')
    )
    offense_formations['% Pass'] = offense_formations['Pass_Plays'] / offense_formations['Plays']

    offense_formations = offense_formations.reset_index().pivot(
        index=['posteam', 'season'],
        columns='OffenseFormation',
        values=['Plays', 'Neutral_Down_Plays', '% Pass']
    ).swaplevel(axis=1)
    offense_formations.columns = [" ".join(col) for col in offense_formations.columns.values]
    offense_formations['% Under Center'] = offense_formations['Under Center Plays'] / (offense_formations['Under Center Plays'] + offense_formations['Shotgun Plays'])
    offense_formations['% Shotgun'] = offense_formations['Shotgun Plays'] / (offense_formations['Under Center Plays'] + offense_formations['Shotgun Plays'])
    offense_formations['% Under Center Neutral Downs'] = offense_formations['Under Center Neutral_Down_Plays'] / (offense_formations['Under Center Neutral_Down_Plays'] + offense_formations['Shotgun Neutral_Down_Plays'])
    offense_formations['% Shotgun Neutral Downs'] = offense_formations['Shotgun Neutral_Down_Plays'] / (offense_formations['Under Center Neutral_Down_Plays'] + offense_formations['Shotgun Neutral_Down_Plays'])

    offense_team_tendencies = offense_team_tendencies.merge(offense_formations, left_index=True, right_index=True, how='left')


    ''' Players - Receiving '''

    # All receivers
    team_targets = pbp_df.groupby(['posteam', 'season', 'receiver']).aggregate(
        Plays=('pass', 'sum'),
        Targets=('pass_attempt', 'sum')
    ).sort_values(by=['posteam', 'season', 'Targets'], ascending=[True, True, False])

    team_targets['Targets'] = pd.to_numeric(team_targets['Targets'])
    team_targets['Target Share'] = team_targets['Targets'] / team_targets.groupby(level=['posteam', 'season'])['Targets'].sum()
    team_targets['Target Share Cumsum'] = team_targets.groupby(level=['posteam', 'season'])['Target Share'].cumsum()
    team_targets['>5% Target Share'] = np.where(team_targets['Target Share'] >= 0.05, 1, 0)

    # Team seasons
    team_targets_seasons = team_targets.groupby(level=['posteam', 'season']).aggregate(
        MaxTargets=('Targets', 'max'),
        MaxTargetShare=('Target Share', 'max'),
        N_Receivers_FivePctTargetShare=('>5% Target Share', 'sum')
    )


    ''' Players - Rushing '''

    # All rushers
    team_rushing = pbp_df.loc[pbp_df['rush'] == 1,:].groupby(['posteam', 'season', 'rusher']).aggregate(
        Plays=('rush', 'sum'),
        Attempts=('rush_attempt', 'sum')
    ).sort_values(by=['posteam', 'season', 'Attempts'], ascending=[True, True, False])

    team_rushing['Attempts'] = pd.to_numeric(team_rushing['Attempts'])
    team_rushing['Attempts Share'] = team_rushing['Attempts'] / team_rushing.groupby(level=['posteam', 'season'])['Attempts'].sum()
    team_rushing['Attempts Share Cumsum'] = team_rushing.groupby(level=['posteam', 'season'])['Attempts Share'].cumsum()
    team_rushing['>10% Attempts Share'] = np.where(team_rushing['Attempts Share'] >= 0.1, 1, 0)

    # Team seasons
    team_rushing_seasons = team_rushing.groupby(level=['posteam', 'season']).aggregate(
        MaxRushAttempts=('Attempts', 'max'),
        MaxRushAttemptsShare=('Attempts Share', 'max'),
        N_Rushers_TenPctAttemptShare=('>10% Attempts Share', 'sum')
    )


    ''' Combine '''

    # Start with base
    offense_inputs = offense_team_tendencies.copy()

    # Add receivers / rushers
    offense_inputs = offense_inputs.merge(team_targets_seasons, left_index=True, right_index=True, how='left')
    offense_inputs = offense_inputs.merge(team_rushing_seasons, left_index=True, right_index=True, how='left')

    # Numeric cols
    for col in offense_inputs.columns:
        offense_inputs[col] = pd.to_numeric(offense_inputs[col])

    # print(offense_inputs.shape)
    # print(offense_inputs.head().to_string())

    return offense_inputs




def load_stats_team_tendencies_defense():
    ''' Prep Defensive Inputs '''
        
    ## Get data ##
    pbp_df = load_pbp_participation_data()
 
    ## Base Stats ##
    defense_team_tendencies = pbp_df.groupby(['defteam', 'season']).aggregate(
        Games=('game_id', 'nunique'),
        Drives=('DriveID', 'nunique'),
        Plays=('defteam', 'size'),
        Neutral_Down_Plays=('defteam', lambda x: x[pbp_df['NeutralDown'] == 1].shape[0]),

        PassPlaysFaced=('pass', 'sum'),
        RushPlaysFaced=('rush', 'sum'),

        LightBoxPlays=('LightBox', 'sum'),
        HeavyBoxPlays=('HeavyBox', 'sum'),
        ZoneCoveragePlays=('ZoneCoverage', 'sum'),
        ManCoveragePlays=('ManCoverage', 'sum'),

        FiveRushersPlays=('pass', lambda x: x[pbp_df['number_of_pass_rushers'] == 5].sum()),
        SixPlusRushersPlays=('pass', lambda x: x[pbp_df['number_of_pass_rushers'] >= 6].sum()),
    )

    # Overall numbers
    defense_team_tendencies['Plays / Game'] = defense_team_tendencies['Plays'] / defense_team_tendencies['Games']
    defense_team_tendencies['Drives / Game'] = defense_team_tendencies['Drives'] / defense_team_tendencies['Games']

    # Box
    defense_team_tendencies['% Light Box'] = defense_team_tendencies['LightBoxPlays'] / defense_team_tendencies['Plays']
    defense_team_tendencies['% Heavy Box'] = defense_team_tendencies['HeavyBoxPlays'] / defense_team_tendencies['Plays']

    # Coverage
    defense_team_tendencies['% Zone'] = defense_team_tendencies['ZoneCoveragePlays'] / defense_team_tendencies['PassPlaysFaced']
    defense_team_tendencies['% Man'] = defense_team_tendencies['ManCoveragePlays'] / defense_team_tendencies['PassPlaysFaced']

    # Rushers
    defense_team_tendencies['% 5 Rushers'] = defense_team_tendencies['FiveRushersPlays'] / defense_team_tendencies['PassPlaysFaced']
    defense_team_tendencies['% 6+ Rushers'] = defense_team_tendencies['SixPlusRushersPlays'] / defense_team_tendencies['PassPlaysFaced']

    # Coverages
    defense_coverages = pbp_df.groupby(['defteam', 'season', 'DefenseCoverage']).aggregate(
        Plays=('pass', 'sum'),
        Neutral_Down_Plays=('pass', lambda x: x[pbp_df['NeutralDown'] == 1].sum()),    
    )
    defense_coverages['% Plays'] = defense_coverages['Plays'] / defense_coverages.groupby(level=['defteam', 'season'])['Plays'].sum()
    defense_coverages['% Neutral Down Plays'] = defense_coverages['Neutral_Down_Plays'] / defense_coverages.groupby(level=['defteam', 'season'])['Neutral_Down_Plays'].sum()

    # Filter out other AFTER calculating percentages
    defense_coverages = defense_coverages.loc[defense_coverages.index.get_level_values('DefenseCoverage') != 'Other',:]
    defense_coverages = defense_coverages.reset_index().pivot(
        index=['defteam', 'season'],
        columns='DefenseCoverage',
        values=defense_coverages.columns
    ).swaplevel(axis=1)
    defense_coverages.columns = [" ".join(col) for col in defense_coverages.columns.values]

    defense_team_tendencies = defense_team_tendencies.merge(defense_coverages, left_index=True, right_index=True)

    print(defense_team_tendencies.shape)
    print(defense_team_tendencies.head().to_string())
    # print(defense_team_tendencies.loc[defense_team_tendencies.index.get_level_values('season') == 2024, :].to_string())


    return defense_team_tendencies
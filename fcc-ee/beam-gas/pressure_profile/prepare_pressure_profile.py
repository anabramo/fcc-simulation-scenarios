import re
import json
from scipy.interpolate import interp1d, Akima1DInterpolator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xtrack as xt
import xpart as xp


def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))


def extract_items_by_regex(dictionary, regex_pattern):
    return {key: value for key, value in dictionary.items() if re.match(regex_pattern, key)}


def load_xtrack_line(lattice_file):
    with open(lattice_file, 'r') as fid:
        line = xt.Line.from_dict(json.load(fid))
    return line


def load_arc_cell_pressure(arc_xls_file, beam=1):
    arc_df = pd.read_excel(arc_xls_file)
    arc_df.drop(['Unnamed: 7', 'INFO'], axis=1, inplace=True)
    # select beam
    arc_df = arc_df.filter(regex=f'S_m|B{beam}')

    # rename columns to manageable names
    rename_map = {'S_m': 's'}
    for cname in arc_df.columns:
        if cname.startswith('B1'):
            rename_map[cname] = cname.split('_')[1].lower()

    arc_df.rename(columns=rename_map, inplace=True)
    return arc_df


def load_mdi_pressure(mdi_xls_file):
    mdi_df = pd.read_excel(mdi_xls_file, names=['s', 'h2', 'co', 'co2'])

    return mdi_df





def stitch_pressure_profile(arc_df, mdi_df, line, interp_step):

    element_positions = dict(zip(line.element_names, line.get_s_elements()))
    circumference = line.get_length()

    # extract the positions of the and separate out the MDI pressure profiles
    ip_pos = extract_items_by_regex(element_positions, '^ip\.\d+$')

    mdi_df_pos = mdi_df[mdi_df['s'] > 0]
    mdi_df_neg = mdi_df[mdi_df['s'] < 0]

    # ip 1 has two  markers - one at azeo and one at the end of the line
    ip_names = [f'ip_{i}' for i in range(1, 5)]
    ip_s = np.array(sorted(list(set(ip_pos.values()))))[:-1]
    ip_s_dict = dict(zip(ip_names, ip_s))

    # extract the arcs from the alttice
    # marker families delimiting the arcs
    start_arc_ip_to_tech = extract_items_by_regex(element_positions, '^fd0a\.\d+$')
    end_arc_ip_to_tech = extract_items_by_regex(element_positions, '^fd3a\.\d+$')
    start_arc_tech_to_ip = extract_items_by_regex(element_positions, '^ff3\.\d+$')
    end_arc_tech_to_ip = extract_items_by_regex(element_positions, '^ff1\.\d+$')

    ip_to_tech_arc_names = ['arc_AB', 'arc_DF', 'arc_GH', 'arc_JL']
    # some of the marker names are duplicated in pairs at the same position and the
    # numerical indices are confising, so remove duplucates and sort on the s
    arc_ip_to_tech_start_s = np.array(sorted(list(set(start_arc_ip_to_tech.values()))))
    arc_ip_to_tech_end_s = np.array(sorted(list(set(end_arc_ip_to_tech.values()))))
    ip_to_tech_arc_s_dict = dict(zip(ip_to_tech_arc_names, zip(arc_ip_to_tech_start_s, arc_ip_to_tech_end_s)))

    tech_to_ip_arc_names = ['arc_BD', 'arc_FG', 'arc_HJ', 'arc_LA']
    arc_tech_to_ip_start_s = np.array(sorted(list(set(start_arc_tech_to_ip.values()))))
    arc_tech_to_ip_end_s = np.array(sorted(list(set(end_arc_tech_to_ip.values()))))
    tech_to_ip_arc_s_dict = dict(zip(tech_to_ip_arc_names, zip(arc_tech_to_ip_start_s, arc_tech_to_ip_end_s)))


    quadrants_s_dict = {
        1: [ip_to_tech_arc_s_dict['arc_AB'], tech_to_ip_arc_s_dict['arc_BD']],
        2: [ip_to_tech_arc_s_dict['arc_DF'], tech_to_ip_arc_s_dict['arc_FG']],
        3: [ip_to_tech_arc_s_dict['arc_GH'], tech_to_ip_arc_s_dict['arc_HJ']],
        4: [ip_to_tech_arc_s_dict['arc_JL'], tech_to_ip_arc_s_dict['arc_LA']],
    }

    # All the same length, so just take the first
    tech_length = list(tech_to_ip_arc_s_dict.values())[0][0] - list(ip_to_tech_arc_s_dict.values())[0][1]
    # Dummy df to zero out the pressure in the technical insertions
    n_dummy_points = 100
    tech_df_dummy = pd.DataFrame({kk: np.full(n_dummy_points, 0.0) for kk in arc_df.columns})
    tech_ins_margin = 0.1 # give some maring to fit trucnated arc cells
    tech_df_dummy['s'] = np.linspace(tech_ins_margin, tech_length - tech_ins_margin, 100)

    inter_ip_distance = ip_s[1] - ip_s[0]
    mdi_pressure_span = mdi_df['s'].max() - mdi_df['s'].min()
    arc_cell_pressure_span = arc_df['s'].max() - arc_df['s'].min()


    quadrant_dfs = []
    for quadrant_index in range(1, 5):
        # The MDI pressure is about the 0 point at the IP
        # Split the MDI pressure profiles into two halves and transform to ring s
        ips = ip_s_dict[f'ip_{quadrant_index}']
        _mdi_upstream = mdi_df_neg.copy()
        _mdi_downstream = mdi_df_pos.copy()
        _mdi_upstream['s'] = (_mdi_upstream['s'] + ips) % circumference
        _mdi_downstream['s'] = (_mdi_downstream['s'] + ips) % circumference

        _mdi_df = pd.concat([_mdi_upstream, _mdi_downstream])


        _arc_parts = []
        for ii, arc_span in enumerate(quadrants_s_dict[quadrant_index]):
            total_arc_pressure_span = arc_span[1] - arc_span[0]
            arc_cell_count = int(total_arc_pressure_span / arc_cell_pressure_span)
            partial_cell_length = total_arc_pressure_span % arc_cell_pressure_span
            partial_cell_df = arc_df[arc_df['s'] < arc_df['s'].min() + partial_cell_length]

            _arc_pressures = [arc_df.copy() for _ in range(arc_cell_count)] + [partial_cell_df.copy()]

            for i in range(arc_cell_count + 1): # the +1 is for the partial last cell
                _arc_pressures[i]['s'] = _arc_pressures[i]['s'] + arc_span[0] + i * arc_cell_pressure_span
            _arc_df = pd.concat(_arc_pressures).reset_index(drop=True)
            _arc_parts.append(_arc_df)

            # The tech insertion is placed at the end of the first arc in the quadrant
            if ii == 0:
                tech_ins = tech_df_dummy.copy()
                tech_ins['s'] = tech_ins['s'] + arc_span[1]
                _arc_parts.append(tech_ins)

        _arc_quadrant_df = pd.concat(_arc_parts).reset_index(drop=True)
        _arc_quadrant_df.sort_values('s', inplace=True)
        _full_quadrant_df = pd.concat([_mdi_df, _arc_quadrant_df]).reset_index(drop=True)
        quadrant_dfs.append(_full_quadrant_df)


    full_df = pd.concat(quadrant_dfs)
    full_df.sort_values('s', inplace=True)
    full_df.reset_index(drop=True)
    full_df.drop_duplicates(subset=['s'], inplace=True)
    

    sample_s_position = np.arange(full_df.s.min(), full_df.s.max(), interp_step)
    interp_dict = {}
    interp_dict['s'] = sample_s_position
    for col in list(set(full_df.columns) - {'s'}):
        interp_dict[col] = interp1d(full_df['s'], full_df[col])(sample_s_position)
    interp_df = pd.DataFrame(interp_dict)

    from IPython import embed; embed()

    
    fig, ax = plt.subplots()
    ax.plot(full_df.s, full_df.h2, lw=0.8, label='h2', zorder=10)
    ax.plot(interp_df.s, interp_df.h2, lw=0.7, label=f'h2 interp {interp_step:.2f} m', zorder=10)
    # ax.plot(full_df.s, full_df.co, label='co', zorder=9)
    # ax.plot(full_df.s, full_df.co2, label='co2', zorder=8)
    ax.set_xlabel('s [m]')
    ax.set_ylabel('P [mbar]')
    ax.legend(loc='upper right').set_zorder(100)
    # plt.savefig('pressure_profile.png', dpi=600, bbox_inches='tight')

    print(f'avg interp h2 {interp_step:.2f} / avg nominal', np.mean(interp_df.h2) / np.mean(full_df.h2))

    plt.show()


def main():
    arc_xls_file = '../input_molflow/FCCee_arcs_GBroggi.xlsx'
    mdi_xls_file = '../input_molflow/FCCee_MDI_pressures_4GiacomoBroggi.xlsx'
    line_file = '../lattice/FCCee-z-collv3-sr-line.json'

    arc_df = load_arc_cell_pressure(arc_xls_file, beam=1)
    mdi_df = load_mdi_pressure(mdi_xls_file)

    line = load_xtrack_line(line_file)

    stitch_pressure_profile(arc_df, mdi_df, line, 1)

if __name__ == '__main__':
    main()

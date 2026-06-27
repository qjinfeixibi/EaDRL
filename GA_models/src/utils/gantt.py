                     

                                                                    
                                                                         

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import colors as mcolors

colors = []

for name, hex in mcolors.cnames.items():
    colors.append(name)
                           

def get_makespan(ma_data):
    max_len = []

                               
    for machine, operations in sorted(ma_data.items()):
                                                                       
        for op in operations:
            max_len.append(op[1])
    return max(max_len)


def parse_data(file):
    try:
        textlist = open(file).readlines()
    except:
        return

    data = {}

    for tx in textlist:
        if not tx.startswith('#'):
            splitted_line = tx.split(',')
            machine = splitted_line[0]
            operations = []

            for op in splitted_line[1::]:
                label = op.split(':')[0].strip()
                l = op.split(':')[1].strip().split('-')
                start = int(l[0])
                end = int(l[1])
                operations.append([start, end, label])

            data[machine] = operations
    return data


def draw_chart(ma_data, veh_data):
    nb_row = len(ma_data.keys())
    veh_nb_row = len(veh_data.keys())

    pos = np.arange(0.5, nb_row * 0.5 + 0.5, 0.5)
    veh_pos = np.arange(0.5, veh_nb_row * 0.5 + 0.5, 0.5)

                                        
                                
                                
    fig, axs = plt.subplots(2, 1, figsize=(20,14), sharex=True)
    ax1 = axs[0]
    ax2 = axs[1]

    index = 0
    veh_index = 0
    max_len = []

                               
    for machine, operations in sorted(ma_data.items()):
                                                                       
        for op in operations:
            max_len.append(op[1])
                                       
            job_idx = int(op[2].split('-')[0])
                                         
            c = colors[job_idx]
            rect = ax1.barh((index * 0.5) + 0.5, op[1] - op[0], left=op[0], height=0.3, align='center',
                           edgecolor=c, color=c, alpha=0.8)

                          
            width = int(rect[0].get_width())
            Str = "OP_{}".format(op[2])
            xloc = op[0] + 0.50 * width
            clr = 'black'
            align = 'center'

            yloc = rect[0].get_y() + rect[0].get_height() / 2.0
            ax1.text(xloc, yloc, Str, horizontalalignment=align,
                            verticalalignment='center', color=clr, weight='bold',
                            clip_on=True)
        index += 1

                               
    for vehicle, operations in sorted(veh_data.items()):
                                                
        for op in operations:
                                       
            job_idx = int(op[2].split('-')[0])
            c = colors[job_idx]
            rect = ax2.barh((veh_index * 0.5) + 0.5, op[1] - op[0], left=op[0], height=0.3, align='center',
                           edgecolor=c, color=c, alpha=0.8)

                          
            width = int(rect[0].get_width())
            Str = "{}".format(op[2])
            xloc = op[0] + 0.50 * width
            clr = 'black'
            align = 'center'

            yloc = rect[0].get_y() + rect[0].get_height() / 2.0
            ax2.text(xloc, yloc, Str, horizontalalignment=align,
                            verticalalignment='center', color=clr, weight='bold',
                            clip_on=True)
        veh_index += 1

    ax1.set_ylim(ymin=-0.1, ymax=nb_row * 0.5 + 0.5)
    ax1.grid(color='gray', linestyle=':')
    ax1.set_xlim(0, max(10, max(max_len)))
    
    ax2.set_ylim(ymin=-0.1, ymax=veh_nb_row * 0.5 + 0.5)
    ax2.grid(color='gray', linestyle=':')
    ax2.set_xlim(0, max(10, max(max_len)))

                                     
                                                
                                                      
                                    
    ax1.set_yticks(pos)
    ax1.set_yticklabels(ma_data.keys(), fontsize=14)

                                     
                                                
                                                           
                                    
    ax2.set_yticks(veh_pos)
    ax2.set_yticklabels(veh_data.keys(), fontsize=14)
    
                                                      
                                  
                                  

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    

    plt.title("Flexible Job Shop Solution")
    plt.savefig('gantt.svg')
    plt.show()


def export_latex(data):
    max_len = []
    head = """
\\noindent\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tikzpicture}}[x=.5cm, y=1cm]
\\begin{{ganttchart}}{{1}}{{{}}}
[vgrid, hgrid]{{{}}}
\\gantttitle{{Flexible Job Shop Solution}}{{{}}} \\\\
\\gantttitlelist{{1,...,{}}}{{1}} \\\\
"""
    footer = """
\\end{ganttchart}
\\end{tikzpicture}}\n
    """
    body = ""
    for machine, operations in sorted(data.items()):
                                                                       
        counter = 0
        for op in operations:
            max_len.append(op[1])
            label = "O$_{{{}}}$".format(op[2].replace('-', ''))
                                                                                                  
            body += "\\Dganttbar{{{}}}{{{}}}{{{}}}{{{}}}".format(machine, label, op[0], op[1])
            if counter == (len(operations) - 1):
                body += "\\\\ \n"
            else:
                body += "\n"
            counter += 1

    lenM = max(10, max(max_len))
    print(head.format(lenM, lenM, lenM, lenM))
    print(body)
    print(footer)


if __name__ == '__main__':
    fname = r"test.txt"
    draw_chart(parse_data(fname))

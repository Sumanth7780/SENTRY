from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import seaborn as sns
import io
import base64
import logging
import os
app = Flask(__name__)
app.logger.setLevel(logging.INFO)




# Initialization functions
class instance_nLY:
    def __init__(self, s=[], beta=[], alpha=[], theta=[], gamma=[], cost=[], C_bar=[]):
        self.s = s
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.cost = cost
        self.C_bar = C_bar

    def print_values(self):
        print("s =", self.s)
        print("beta =", self.beta)
        print("alpha =", self.alpha)
        print("theta =", self.theta)
        print("gamma =", self.gamma)
        print("cost =", self.cost)
        print("C_bar =", self.C_bar)

def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def objective_prob(Y, _nLayers):
    global obj2
    f = [None] * _nLayers
    for i in range(len(f)):
        f[i] = np.exp(-1 * sum([obj2.gamma[i-k] * obj2.theta[k] * Y[k] for k in range(i+1)]))
    raw_risk = [_s * _beta * _alpha for _s, _beta, _alpha in zip(obj2.s, obj2.beta, obj2.alpha)]
    return sum(a * b for a, b in zip(raw_risk, f))

def objective_stra(Y, _nLayers):
    global obj2
    f = [None] * _nLayers
    for i in range(len(f)):
        f[i] = np.exp(-1 * sum([obj2.gamma[i-k] * obj2.theta[k] * Y[k] for k in range(i+1)]))
    raw_risk = [_s * _alpha for _s, _alpha in zip(obj2.s, obj2.alpha)]
    return max(a * b for a, b in zip(raw_risk, f))

def constraint(Y, obj2):
    return obj2.C_bar - sum(a * b for a, b in zip(obj2.cost, Y))

def get_numerical_sol(init_val, _model, _nLayers, obj2):
    Y0 = init_val
    b = (0.0, None)
    bnds = (b,) * _nLayers
    con1 = {'type': 'eq', 'fun': lambda Y: constraint(Y, obj2)}
    cons = ([con1])
    if _model == 'prob':
        solution = minimize(lambda Y: objective_prob(Y, _nLayers), Y0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        x = [round(i, 4) for i in x]
        obj_value = round(objective_prob(x, _nLayers), 4)
    elif _model == 'stra':
        solution = minimize(lambda Y: objective_stra(Y, _nLayers), Y0, method='SLSQP', bounds=bnds, constraints=cons)
        x = solution.x
        x = [round(i, 4) for i in x]
        obj_value = round(objective_stra(x, _nLayers), 4)
    else:
        print("Something is wrong here......(get_numerical_sol)")
    return flatten_list([obj_value, x])

def addRow(df, ls):
    numEl = len(ls)
    newRow = pd.DataFrame(np.array(ls).reshape(1, numEl), columns=list(df.columns))
    df = pd.concat([df, newRow], ignore_index=True)
    return df

def get_full_sol(_nLayers, whichmodel, obj2, vars_col):
    intial_sol = [3] * _nLayers
    solutions = get_numerical_sol(intial_sol, whichmodel, _nLayers, obj2)
    required_length = len(vars_col)
    while len(solutions) < required_length:
        solutions.append(0)
    return solutions[:required_length]

def initialization(_nLayers, C_bar_init):
    gam = 0.5
    s_init = [500 for i in range(1, _nLayers+1)]
    alpha_init = [0.5] * _nLayers
    beta_init = [1 / _nLayers] * _nLayers
    theta_init = [0.4] * _nLayers
    cost_init = [1 for i in range(1, _nLayers+1)]
    gamma_init = [1] + [gam**i for i in range(1, _nLayers)]
    obj_base = instance_nLY(s=s_init, alpha=alpha_init, beta=beta_init, theta=theta_init, cost=cost_init, gamma=gamma_init, C_bar=C_bar_init)
    return obj_base

def generate_plot(whichmodel, solution_df, total_layers, obj2):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    global obj2

    if request.method == 'POST':
        model_type = request.form.get('model_type')
        total_layers = int(request.form.get('total_layers'))
        C_bar_init = float(request.form.get('C_bar_init'))

        app.logger.info(f"Total Layers: {total_layers}")
        app.logger.info(f"C_bar_init: {C_bar_init}")

        vars_col = ['obj_value'] + ['y' + str(i+1) for i in range(total_layers)]
        model_set = ['prob', 'stra']
        prob_solutions = None
        stra_solutions = None

        labels = ['Layer ' + str(i+1) for i in range(total_layers)]
        plot_urls = {}

        # Select the appropriate image based on total_layers
        layer_image = None
        if total_layers in [1, 2, 3, 4]:
            layer_image = f"layer{total_layers}.png"

        for m in range(len(model_set)):
            whichmodel = model_set[m]
            solution_df = pd.DataFrame(columns=vars_col)

            for i in range(total_layers):
                _nLayers = i + 1
                obj_base = initialization(_nLayers, C_bar_init)
                obj2 = copy.deepcopy(obj_base)
                solutions = get_full_sol(_nLayers, whichmodel, obj2, vars_col)
                solution_df = addRow(solution_df, solutions)
            
            solution_df["Layers"] = [i for i in range(1, total_layers+1)] 

            if whichmodel == 'prob':
                prob_solutions = solutions
            elif whichmodel == 'stra':
                stra_solutions = solutions
            else:
                print("Something is wrong.")

            finaldata = pd.DataFrame()
            for i in range(0, total_layers):
                new_name = 'invest' + str(i+1)
                old_name = 'y' + str(i+1)
                finaldata[new_name] = obj2.cost[i] * solution_df[old_name]

            data_perc = 100 * finaldata[finaldata.columns.values.tolist()].divide(finaldata[finaldata.columns.values.tolist()].sum(axis=1), axis=0)
            data_perc["Layers"] = [i for i in range(1, total_layers+1)]

            for i in range(1, total_layers + 1):
                col = 'invest' + str(i)
                data_perc[col] = pd.to_numeric(data_perc[col], errors='coerce')

            plt.stackplot(data_perc['Layers'],
                          [data_perc['invest' + str(i+1)] for i in range(total_layers)],
                          labels=labels,
                          colors=sns.color_palette(("Greys"), total_layers),
                          alpha=1, edgecolor='grey')

            plt.margins(x=0)
            plt.margins(y=0)
            plt.rcParams['font.size'] = 15
            if whichmodel == 'prob':
                plt.ylabel("Budget allocation (%)", fontsize=18)
                plt.xlabel("Number of layers for LRA-PR Model", fontsize=18)
            if whichmodel == 'stra':
                plt.xlabel("Number of layers for LRA-SR Model", fontsize=18)

            plt.legend(fontsize=14, markerscale=2, loc='center left', bbox_to_anchor=(1.2, 0.5), fancybox=True, shadow=True, ncol=1)

            plot_urls[whichmodel] = generate_plot(whichmodel, solution_df, total_layers, obj2)

        if model_type == 'prob':
            display_solutions = prob_solutions
            display_model = 'prob'
        else:
            display_solutions = stra_solutions
            display_model = 'stra'

        return render_template(
            'results_template.html',
            model_type=model_type,
            plot_url=plot_urls[display_model],
            total_layers=total_layers,
            C_bar_init=C_bar_init,
            model=display_model,
            solutions=str(display_solutions),
            s=str(obj2.s),
            beta=str(obj2.beta),
            alpha=str(obj2.alpha),
            theta=str(obj2.theta),
            gamma=str(obj2.gamma),
            cost=str(obj2.cost),
            C_bar=str(obj2.C_bar),
            layer_image=layer_image, consequence="0.5", vulnerability="0.56", threat="0.5")  # New parameter for layer-specific image

    return render_template('form_template.html')

@app.route('/about')
def about():
    return render_template('about_template.html')

@app.route('/contact')
def contact():
    return render_template('contact_template.html')

@app.route('/results', methods=['POST'])
def results():
    return index()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

#if __name__ == "__main__":
#    app.run(debug=True, port=5000)

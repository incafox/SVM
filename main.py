import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Vectores_soporte:
    def __init__(self, visualizacion=True):
        self.visualizacion = visualizacion
        self.colores = {1:'r',-1:'b'}
        if self.visualizacion:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    # entrena
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # vectores de soporte yi(xi.w+b) = 1
        
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # punto de expancion:
                      self.max_feature_value * 0.001,]    
        
        # #
        b_range_multiple = 5
        # multiplo 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # se puede por ser convexo
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # SMO llega a arreglar un poco esto
                        # yi(xi.w+b) >= 1
                        # 
                        # el break luego
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print('Optimizado un poco.')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
    def predict(self,features):
        # sign( x.w+b )
        clasificacion = np.sign(np.dot(np.array(features),self.w)+self.b)
        return clasificacion
        
datos = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.figure as mpfig
import seaborn as sns

class Matcher(object):
    """docstring for Match"""
    def __init__(self, obj_win, search_img):
        super(Matcher, self).__init__()
        self._obj_win = obj_win
        self._search_img = search_img

    def get_best_coords_matched(self, criteria = 'corr_func'):
        '''从 ._search_img 中 找出搜索窗口在 'critera' 标准下最好
        的坐标.
        请注意: 比较时 最好的 得分最高,因此在编写 criteria函数时候,
        越好的搜索窗口返回的得分是最高的.避免在某个标准下好的窗口得分
        低.比如，差平方和 标准
        返回 匹配最好的坐标(x, y)
        '''
        m, n = self._search_img.shape
        best_x, best_y = 0, 0
        best_value = float('-inf')

        for i in range(1, m -2):
            for j in range(1, n - 2):
                search_win = np.copy( self._search_img[i-1:i+2, j-1:j+2] )
                if criteria == 'corr_func':
                    score = self.correlation_func(self._obj_win, search_win)
                elif criteria == 'corr_effi':
                    score = self.correlation_effi(self._obj_win, search_win)
                elif criteria == 'cov_func':
                    score = self.covariance_func(self._obj_win, search_win)
                elif criteria == 'mod1':
                    score = self.mod1_of_diffrence(self._obj_win, search_win)
                elif criteria == 'mod2':
                    score = self.mod2_of_difference(self._obj_win, search_win)
                else:
                    raise ValueError("parameter error, %s not wrotten"%criteria)

                if score > best_value:
                    best_x, best_y = i, j
                    best_value = score
        return best_x, best_y

    def correlation_func(self, obj_win, search_win):
        """Correlation fucntion"""
        s = np.multiply(obj_win, search_win).sum()
        return s 

    def covariance_func(self, obj_win, search_win):
        '''Covariance function '''
        assert obj_win.shape == search_win.shape
        m, n = obj_win.shape
        obj_win = obj_win - obj_win.sum() / (m * n)
        search_win = search_win - search_win.sum() / (m * n) 

        s = np.multiply(obj_win, search_win).sum()
        return s 

    def correlation_effi(self, obj_win, search_win):
        '''Correlation efficient'''
        assert obj_win.shape == search_win.shape, "windows' shape not matched"
        C = self.covariance_func(obj_win, search_win)
        ro = C / (obj_win.std() * search_win.std() )
        return abs(ro)

    def mod2_of_difference(self, obj_win, search_win):
        '''sum of the square of the difference of two vector'''
        assert obj_win.shape == search_win.shape, "windows' shape not matched"
        diff = (obj_win - search_win).flatten()
        return - np.dot(diff, diff) #

    def mod1_of_diffrence(self, obj_win, search_win):
        '''sum of the absolute value of the difference of 2 vectors'''
        assert obj_win.shape == search_win.shape, "windows' shape not matched"
        diff = (obj_win - search_win).flatten()

        return -abs(diff).sum()# some comments

class MatcherTester(object):
    """docstring for MatcherTester"""
    def __init__(self, obj_points_fname = 'data/1BAK_points.txt',
                obj_img_fname = 'data/1BAK.bmp', 
                search_img_fname = 'data/2BAK.bmp'):
        
        super(MatcherTester, self).__init__()
        self._obj_points_fname = obj_points_fname
        self._obj_img_fname = obj_img_fname
        self._search_img_fname = search_img_fname
        
        self._obj_img = mpimg.imread(self._obj_img_fname)
        self._search_img = mpimg.imread(self._search_img_fname)

        self.get_points()
    
    def get_points(self):
        """从文件中读取 目标点的坐标"""
        f = open(self._obj_points_fname)
        ob = f.readlines()
        # 前后有可能有空格
        ob = [l for l in ob if len(l)>5]

        ob = [l.split() for l in ob]

        m, n = np.array(ob).shape
        for i in range(m):
            for j in range(n):
                ob[i][j] = int(ob[i][j])
        ob = np.array(ob)
        self._coord_name = ob[:, 0]
        self._coord_poi = ob[:, 1:3]
        
    def matched_coords(self, criteria = 'corr_func'):

        m = len(self._coord_name)
        best_pois = []
        for i in range(m): # 对每一个点,构建一个matcher类
            poi_x, poi_y = self._coord_poi[i]
            print(poi_x)
            print(poi_y)
            a = Matcher(self._obj_img[poi_x-1:poi_x+2, poi_y-1:poi_y+2],
                        self._search_img)
            best_poi = a.get_best_coords_matched(criteria)
            best_pois.append(best_poi)
        return best_pois
    def plot_matched_points(self, criteria = 'corr_func'):

        best_pois = self.matched_coords(criteria)
        print(best_pois)

        # fig = mpfig.Figure()
        # plt.imshow(self._search_img, cmap = 'Greys_r')
        # for p, i in zip(best_pois, range(len(best_pois))):
        #     plt.scatter(p[1], p[0]) 
        #     plt.annotate(s = str(self._coord_name[i]), xy=(p[1], p[0]))       
        # plt.show()
        # plt.savefig('data/matchedpic_' + criteria+'_.bmp')      
        plt.figure()
        plt.imshow(self._search_img, cmap = 'Greys_r')
        for p, i in zip(best_pois, range(len(best_pois))):
            plt.scatter(p[1], p[0]) 
            plt.annotate(s = str(self._coord_name[i]), xy=(p[1], p[0]))       
        
        plt.savefig('data/matchedpic_' + criteria+'_.jpg')  
        # savefig before show() !!!
        #  plt.show()

def main():
    criteria_list = ['corr_func', 'corr_effi', 'cov_func', 'mod1', 'mod2']
    
    a = MatcherTester()
    for c in criteria_list:
        a.plot_matched_points(c)

if __name__ == '__main__':
    main()

        

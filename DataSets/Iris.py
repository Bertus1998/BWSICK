class Iris:

    def get_input(self):
        return [self.v1, self.v2, self.v3, self.v4]

    def get_output(self):
        res = [0, 0, 0]
        res[self.type] = 1
        return res

    @staticmethod
    def get_irises(path):
        irises = []
        lines = open(path, 'r').read().split('\n')

        for s in lines[:-2]:
            values = s.split(',')
            i = Iris()
            i.v1 = float(values[0])
            i.v2 = float(values[1])
            i.v3 = float(values[2])
            i.v4 = float(values[3])

            if values[4] == 'Iris-setosa':
                i.type = 0
            elif values[4] == 'Iris-versicolor':
                i.type = 1
            else:
                i.type = 2

            irises.append(i)

        return irises

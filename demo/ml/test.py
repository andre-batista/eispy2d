


class Train:

    def __init__(self):
        self.chain =  {}

    # scf -> [nm, ns]
    # idf -> [N, ns]
    # gs -> [nm, N]
    # gd -> [N, N]
    # chi -> [nx, ny]
    # N = nx * ny
    # nx, ny = (x, y) * 2/3
    def train(self, scf, idf, gs, gd, chi):
        nx, ny = chi.shape
        N = nx * ny
        nm, ns = scf.shape

        for px in range(0, nx):
            for py in range(0, ny):
                incident_mean = mean(idf[px*py])
                for ps in range(0, ns):
                    scattered_mean = mean(scf[][ps])
                    key = (incident_mean, scattered_mean)
                    if key in self.chain:
                        self.chain[key] += chi[px][py]
                        self.chain[key] /= 2
                    else:
                        self.chain[key] = chi[px][py]

    def save(self):
        pass

    def read(self):
        pass


    def solve(self, scf, idf, gs, gd, res):
        nx, ny = res
        chi = np.real()

        for px in range(0, nx):
            for py in range(0, ny):   
                incident_mean = mean(idf[px*py])
                for ps in range(0, ns):
                    scattered_mean = mean(scf[][ps])
                    key = (incident_mean, scattered_mean)
                    pchi = 0 + 0j
                    div = 0
                    for k in self.chain.keys():
                        di = distance(self.chain[k][0], key[0])
                        ds = distance(self.chain[k][1], key[1])
                        pchi += self.chain[k] / (di * ds)
                        div += (di * ds)
                    pchi *= div
                chi[px][py] = pchi

        return sct, chi
                        
    def distance(x, y):
        rd = y.real - x.real
        imd = y.imag - x.imag

        dis = math.sqrt(rd**2 + imd**2)

        return dis







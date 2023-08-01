import math, time
from numpy import *
from functools import reduce

def arraymedian(input, axis="both", lthreshold=None, hthreshold=None, nlow=0, nhigh=0, nonzero=False):
    n = input.size
    doReject = (lthreshold is not None or hthreshold is not None or nonzero)
    if (n == 0): return 0
    if (axis == "both"):
        t = time.time()
        data = reshape(input, n)+0
        if (nonzero):
            dmin = data.min()
            data[data == 0] = dmin-2
            if (lthreshold is None):
                lthreshold = dmin-1
            else:
                lthreshold = maximum(lthreshold, dmin-1)
        data.sort()
        if (doReject):
            if (lthreshold is not None):
                nl = (data < lthreshold).sum()
            else:
                nl = 0
            if (hthreshold is not None):
                nh = (data <= hthreshold).sum()
            else:
                nh = n
            #minmax rejection
            if (nlow > 0 or nhigh > 0):
                nh -= int((nhigh+0.)*(nh-nl)/n)
                nl += int((nlow+0.)*(nh-nl)/n)
            m = (nh-nl)//2+nl
            if (m == n):
                return 0
            if ((nh-nl) % 2 == 1):
                return data[m]
            else:
                return (data[m]+data[m-1])/2.0
        n = n+nlow-nhigh
        #print time.time()-t
        if (n%2 == 1): return data[int(n/2)]
        else: return (data[n//2] + data[n//2-1])/2.0
    elif (axis == "Y"):
        n = input.shape[0]
        sz = input.shape
        #swap axes and copy for faster sort
        data = swapaxes(input,0,1)+0
        if (nonzero):
            dmin = data.min()
            data[data == 0] = dmin-2
            if (lthreshold is None):
                lthreshold = dmin-1
            else:
                lthreshold = maximum(lthreshold, dmin-1)
        data.sort(1)
        if (doReject):
            if (lthreshold is not None):
                if (len(data.shape) > 2):
                    nl = reduce(add, swapaxes(data, 0, 1) < lthreshold, 0).astype("int8")
                else:
                    nl = add.reduce((data < lthreshold)+0, 1)
            else:
                if (len(data.shape) > 2):
                    nl = zeros(swapaxes(data,0,1).shape[1:], "Int8")
                else:
                    nl = zeros(data.shape[1:], "Int8")
            if (hthreshold is not None):
                if (len(data.shape) > 2):
                    nh = reduce(add, swapaxes(data, 0, 1) <= hthreshold, 0)
                else:
                    nh = add.reduce((data <= hthreshold)+0, 1)
            else:
                nh = n
            #minmax rejection
            if (nlow > 0 or nhigh > 0):
                fac = (nh-nl)*(1./n)
                nh -= (nhigh*fac).astype("int8")
                nl += (nlow*fac).astype("int8")
            m = (nh-nl)//2+nl
            mod = (nh-nl+1)%2
            rej = m != n
            nrej = rej.sum()
            if (nrej < rej.size):
                m *= rej
                if (len(data.shape) > 2):
                    data = data.swapaxes(0,1)+0
                    shp = data.shape
                    data = data.reshape((shp[0], data.size//shp[0]))
                    m = m.reshape(data.size//shp[0])
                    mod = mod.reshape(data.size//shp[0])
                    rej = rej.reshape(data.size//shp[0])
                    n = data.shape[1]
                    return reshape((data[(m, arange(n))] + mod*data[(m-1, arange(n))])*1./(1+mod)*rej, shp[1:])
                n = input.shape[1]
                return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)*rej
            if (len(data.shape) > 2):
                data = data.swapaxes(0,1)+0
                shp = data.shape
                data = data.reshape((shp[0], data.size//shp[0]))
                m = m.reshape(data.size//shp[0])
                mod = mod.reshape(data.size//shp[0])
                n = data.shape[1]
                return reshape((data[(m, arange(n))] + mod*data[(m-1,arange(n))])*1./(1+mod), shp[1:])
            n = input.shape[1]
            return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)
        n = n+nlow-nhigh
        if (n%2 == 1): return data[:,int(n/2)]
        else: return (data[:,n//2] + data[:,n//2-1])/2.0
            #return (data[(m, arange(n))] + mod*data[(m-1,arange(n))])*1./(1+mod)
        #if (n%2 == 1): return data[int(n/2),:]
        #else: return (data[n//2,:] + data[n//2-1,:])/2.0
    elif (axis == "X"):
        n = input.shape[1]
        data = input+0
        if (nonzero):
            dmin = data.min()
            data[data == 0] = dmin-2
            if (lthreshold is None):
                lthreshold = dmin-1
            else:
                lthreshold = maximum(lthreshold, dmin-1)
        data.sort(1)
        if (doReject):
            if (lthreshold is not None):
                nl = add.reduce((data < lthreshold)+0, 1)
            else:
                nl = zeros(data.shape[1:], "Int8")
            if (hthreshold is not None):
                nh = add.reduce((data <= hthreshold)+0, 1)
            else:
                nh = n
            #minmax rejection
            if (nlow > 0 or nhigh > 0):
                nh -= (nhigh*(nh-nl)*(1./n)).astype("int8")
                nl += (nlow*(nh-nl)*(1./n)).astype("int8")
            m = (nh-nl)//2+nl
            mod = (nh-nl+1)%2
            rej = m != n
            nrej = rej.sum()
            n = input.shape[0]
            if (nrej < rej.size):
                m *= rej
                return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)*rej
            return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)
        n = n+nlow-nhigh
        if (n%2 == 1): return data[:,int(n/2)]
        else: return (data[:,n//2] + data[:,n//2-1])/2.0

def arraymedian_na(input, axis="both", lthreshold=False, hthreshold=False):
    if (lthreshold == False and type(lthreshold) == type(0)):
        lthreshold = -0.0000000001
    if (hthreshold == False and type(hthreshold) == type(0)):
        hthreshold = 0.0000000001
    n = input.size()
    if (n == 0): return 0
    if (axis == "both"):
        data = reshape(input, n).copy()
        data.sort()
        if (lthreshold or hthreshold):
            if (lthreshold):
                nl = (data < lthreshold).sum()
            else:
                nl = 0
            if (hthreshold):
                nh = (data <= hthreshold).sum()
            else:
                nh = n
            m = (nh-nl)//2+nl
            if (m == n):
                return 0
            if ((nh-nl) % 2 == 1):
                return data[m]
            else:
                return (data[m]+data[m-1])/2.0
        if (n%2 == 1): return data[int(n/2)]
        else: return (data[n//2] + data[n//2-1])/2.0
    elif (axis == "Y"):
        n = input.shape[0]
        #swap axes and copy for faster sort
        data = swapaxes(input,0,1).copy()
        data.sort(1)
        if (lthreshold or hthreshold):
            if (lthreshold):
                nl = add.reduce((data < lthreshold)+0, 1)
            else:
                nl = 0
            if (hthreshold):
                nh = add.reduce((data <= hthreshold)+0, 1)
            else:
                nh = n
            m = (nh-nl)//2+nl
            mod = (nh-nl+1)%2
            rej = m != n
            nrej = rej.sum()
            if (nrej < rej.size()):
                m *= rej
                if (len(data.shape) > 2):
                    data.swapaxes(0,1)
                    shp = data.shape
                    data = reshape(data, (shp[0], data.size()//shp[0]))
                    m = reshape(m, data.size()//shp[0])
                    mod = reshape(mod, data.size()//shp[0])
                    rej = reshape(rej, data.size()//shp[0])
                    n = data.shape[1]
                    return reshape((data[(m, arange(n))] + mod*data[(m-1, arange(n))])*1./(1+mod)*rej, shp[1:])
                return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)*rej
            if (len(data.shape) > 2):
                data.swapaxes(0,1)
                shp = data.shape
                data = reshape(data, (shp[0], data.size()//shp[0]))
                m = reshape(m, data.size()//shp[0])
                mod = reshape(mod, data.size()//shp[0])
                n = data.shape[1]
                return reshape((data[(m, arange(n))] + mod*data[(m-1,arange(n))])*1./(1+mod), shp[1:])
            return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)
        if (n%2 == 1): return data[:,int(n/2)]
        else: return (data[:,n//2] + data[:,n//2-1])/2.0
            #return (data[(m, arange(n))] + mod*data[(m-1,arange(n))])*1./(1+mod)
        #if (n%2 == 1): return data[int(n/2),:]
        #else: return (data[n//2,:] + data[n//2-1,:])/2.0
    elif (axis == "X"):
        n = input.shape[1]
        data = input.copy()
        data.sort(1)
        if (lthreshold or hthreshold):
            if (lthreshold):
                nl = add.reduce((data < lthreshold)+0, 1)
            else:
                nl = 0
            if (hthreshold):
                nh = add.reduce((data <= hthreshold)+0, 1)
            else:
                nh = n
            m = (nh-nl)//2+nl
            mod = (nh-nl+1)%2
            rej = m != n
            nrej = rej.sum()
            if (nrej < rej.size()):
                m *= rej
                return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)*rej
            return (data[(arange(n), m)] + mod*data[(arange(n), m-1)])*1./(1+mod)
        if (n%2 == 1): return data[:,int(n/2)]
        else: return (data[:,n//2] + data[:,n//2-1])/2.0

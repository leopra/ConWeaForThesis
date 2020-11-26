#here i choose the threshold for choosing verticals
import numpy as np

def vertPicker(countvec):
    meanplus = countvec.mean() +1

    picked = (countvec > meanplus).astype(int)

    xc = list(zip(picked.tolist(), range(len(countvec))))
    xsorted = sorted(xc, key=lambda x: x[0], reverse=True)
    picked = [x for i,x in xsorted if i == 1]
    return picked



    # #TODO this is tricky
    # endcondition = False
    #
    # while (endcondition == False and len(xsorted)>0):
    #     possiblelbl = xsorted.pop()
    #     if possiblelbl == 0:
    #         break
    #     else:
    #         #get the element before
    #         possbefore = xsorted[-1]
    #         #if the one is bigger than the second, add only the first for now
    #         if possiblelbl[0] > possbefore[0]:
    #             picked.append(possiblelbl[1])
    #         #if the second is same as first add both and pop one from xsorted
    #         if possiblelbl[0] == possbefore[0]:
    #             picked.append(possiblelbl[1])
    #             picked.append(possbefore[1])
    #             temp = xsorted.pop()
    #
    #         #if 4 labels canot be more so quit
    #         if len(picked) > 3:
    #             endcondition = True









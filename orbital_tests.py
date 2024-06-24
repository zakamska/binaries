# In this file, we present the use cases of the functions in orbital.py

def test_scal():
    a=np.ones((2,3))
    a[1,2]=5
    b=np.ones((2,3))*2
    b[0,0]=0
    print('a: ', a)
    print('b: ', b)
    print('vector product: ', np.cross(a,b))
    print('scalar product: ', scal(a,b))
    print('projection onto z axis:', a.dot(np.transpose(np.array([0,0,1]))))
    return(1)
    #output:
    #a:  [[1. 1. 1.]
    #     [1. 1. 5.]]
    #b:  [[0. 2. 2.]
    #     [2. 2. 2.]]
    #vector product:  [[ 0. -2.  2.]
    #                  [-8.  8.  0.]]
    #scalar product:  [ 4. 14.]
    #projection onto z axis: [1. 5.]


#generates the .npy file used in the markov chains lab
import numpy as np
A = np.empty((225,225), dtype=bool)
A[:] = False
indices = [(0,15),(1,16),(1,2),(2,3),(3,18),(4,5),
           (5,6),(6,7),(6,21),(7,8),(8,23),(9,10),
           (9,24),(10,11),(11,12),(12,13),(13,14),
           (13,28),(15,30),(16,31),(17,18),(18,19),
           (18,33),(19,20),(20,35),(21,22),(22,37),
           (23,38),(24,39),(25,26),(26,41),(27,28),
           (27,42),(28,29),(29,44),(30,45),(31,32),
           (32,47),(33,34),(35,36),(35,50),(36,37),
           (38,39),(39,40),(40,41),(43,58),(44,59),
           (45,46),(46,61),(47,62),(48,49),(48,63),
           (49,64),(50,65),(51,52),(51,66),(52,67),
           (53,54),(53,68),(54,69),(55,56),(55,70),
           (56,57),(57,58),(58,73),(59,74),(60,61),
           (60,75),(61,62),(63,78),(64,79),(65,66),
           (67,82),(68,83),(69,84),(70,71),(71,72),
           (72,87),(73,88),(74,89),(75,76),(76,77),
           (77,78),(79,80),(80,81),(81,96),(82,83),
           (84,85),(85,86),(86,87),(88,103),(89,104),
           (90,91),(90,105),(91,92),(92,107),(93,94),
           (93,108),(94,109),(95,96),(96,97),(97,112),
           (98,99),(98,113),(99,100),(100,101),(101,102),
           (102,103),(104,119),(105,120),(106,107),(107,108),
           (109,110),(110,111),(110,125),(111,112),(113,128),
           (114,115),(114,129),(115,116),(116,117),(117,118),
           (118,119),(120,121),(121,136),(122,123),(123,124),
           (123,138),(124,125),(126,127),(126,141),(127,142),
           (128,143),(129,144),(130,131),(130,145),(131,132),
           (132,133),(133,134),(134,149),(135,136),(135,150),
           (137,152),(138,139),(138,153),(139,154),(140,155),
           (141,156),(142,157),(143,158),(144,145),(146,147),
           (146,161),(147,148),(148,149),(149,164),(150,151),
           (151,152),(151,166),(153,168),(154,169),(155,170),
           (156,171),(157,172),(158,159),(159,160),(160,161),
           (162,163),(162,177),(163,178),(164,179),(165,166),
           (165,180),(167,168),(167,182),(170,171),(170,185),
           (171,186),(172,173),(173,174),(174,189),(175,176),
           (175,190),(176,191),(177,192),(178,193),(179,194),
           (180,195),(181,182),(181,196),(182,183),(183,198),
           (184,185),(186,201),(187,188),(187,202),(188,189),
           (189,190),(191,192),(195,210),(196,211),(197,198),
           (198,199),(199,200),(200,215),(201,216),(202,217),
           (203,204),(203,218),(204,219),(205,220),(205,206),
           (206,221),(207,222),(208,209),(208,223),(209,224),
           (211,212),(212,213),(213,214),(215,216),(217,218),
           (219,220),(221,222),(222,223)]
A[zip(*indices)] = True
A[:] = A + A.T
np.save("maze.npy", A)

# 数据与参考 PDF 只读审计

本报告仅访问配置指定的路径；未知风险不作否定性结论。

- 数据目录：`C:\Users\k64183565\Desktop\w\数据`（存在：True）
- PDF：`C:\Users\k64183565\Desktop\w\xin\郭论文.pdf`（not_extracted）

## abalone19.dat
- 解析：success；尺寸：252804 字节
- 维度：[4174, 9]；观察到末列：Class；类别：{'negative': 4142, 'positive': 32}
- 非数值列：['Sex', 'Class']；缺失：{'Sex': 0, 'Length': 0, 'Diameter': 0, 'Height': 0, 'Whole_weight': 0, 'Shucked_weight': 0, 'Viscera_weight': 0, 'Shell_weight': 0, 'Class': 0}；完全重复行：0；重复特征行：0
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## ecoli-0-1-3-7_vs_2-6.dat
- 解析：success；尺寸：12273 字节
- 维度：[281, 8]；观察到末列：Class；类别：{'negative': 274, 'positive': 7}
- 非数值列：['Class']；缺失：{'Mcg': 0, 'Gvh': 0, 'Lip': 0, 'Chg': 0, 'Aac': 0, 'Alm1': 0, 'Alm2': 0, 'Class': 0}；完全重复行：0；重复特征行：0
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## ecoli-0-1-4-6_vs_5.dat
- 解析：success；尺寸：11072 字节
- 维度：[280, 7]；观察到末列：class；类别：{'negative': 260, 'positive': 20}
- 非数值列：['class']；缺失：{'a1': 0, 'a2': 0, 'a3': 0, 'a5': 0, 'a6': 0, 'a7': 0, 'class': 0}；完全重复行：0；重复特征行：0
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## ecoli-0-6-7_vs_5.dat
- 解析：success；尺寸：8763 字节
- 维度：[220, 7]；观察到末列：class；类别：{'negative': 200, 'positive': 20}
- 非数值列：['class']；缺失：{'a1': 0, 'a2': 0, 'a3': 0, 'a5': 0, 'a6': 0, 'a7': 0, 'class': 0}；完全重复行：0；重复特征行：0
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## glass-0-1-2-3_vs_4-5-6.dat
- 解析：success；尺寸：18708 字节
- 维度：[214, 10]；观察到末列：Class；类别：{'negative': 163, 'positive': 51}
- 非数值列：['Class']；缺失：{'RI': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 'Si': 0, 'K': 0, 'Ca': 0, 'Ba': 0, 'Fe': 0, 'Class': 0}；完全重复行：1；重复特征行：1
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## glass-0-1-5_vs_2.dat
- 解析：success；尺寸：13463 字节
- 维度：[172, 10]；观察到末列：typeGlass；类别：{'negative': 155, 'positive': 17}
- 非数值列：['typeGlass']；缺失：{'RI': 0, 'Na': 0, 'Mg': 0, 'Al': 0, 'Si': 0, 'K': 0, 'Ca': 0, 'Ba': 0, 'Fe': 0, 'typeGlass': 0}；完全重复行：1；重复特征行：1
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## kddcup-land_vs_satan.dat
- 解析：success；尺寸：242385 字节
- 维度：[1610, 42]；观察到末列：Class；类别：{'negative': 1589, 'positive': 21}
- 非数值列：['Atr-1', 'Atr-2', 'Atr-3', 'Class']；缺失：{'Atr-0': 0, 'Atr-1': 0, 'Atr-2': 0, 'Atr-3': 0, 'Atr-4': 0, 'Atr-5': 0, 'Atr-6': 0, 'Atr-7': 0, 'Atr-8': 0, 'Atr-9': 0, 'Atr-10': 0, 'Atr-11': 0, 'Atr-12': 0, 'Atr-13': 0, 'Atr-14': 0, 'Atr-15': 0, 'Atr-16': 0, 'Atr-17': 0, 'Atr-18': 0, 'Atr-19': 0, 'Atr-20': 0, 'Atr-21': 0, 'Atr-22': 0, 'Atr-23': 0, 'Atr-24': 0, 'Atr-25': 0, 'Atr-26': 0, 'Atr-27': 0, 'Atr-28': 0, 'Atr-29': 0, 'Atr-30': 0, 'Atr-31': 0, 'Atr-32': 0, 'Atr-33': 0, 'Atr-34': 0, 'Atr-35': 0, 'Atr-36': 0, 'Atr-37': 0, 'Atr-38': 0, 'Atr-39': 0, 'Atr-40': 0, 'Class': 0}；完全重复行：685；重复特征行：685
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## kddcup-rootkit-imap_vs_back.dat
- 解析：success；尺寸：344247 字节
- 维度：[2225, 42]；观察到末列：Class；类别：{'negative': 2203, 'positive': 22}
- 非数值列：['Atr-1', 'Atr-2', 'Atr-3', 'Class']；缺失：{'Atr-0': 0, 'Atr-1': 0, 'Atr-2': 0, 'Atr-3': 0, 'Atr-4': 0, 'Atr-5': 0, 'Atr-6': 0, 'Atr-7': 0, 'Atr-8': 0, 'Atr-9': 0, 'Atr-10': 0, 'Atr-11': 0, 'Atr-12': 0, 'Atr-13': 0, 'Atr-14': 0, 'Atr-15': 0, 'Atr-16': 0, 'Atr-17': 0, 'Atr-18': 0, 'Atr-19': 0, 'Atr-20': 0, 'Atr-21': 0, 'Atr-22': 0, 'Atr-23': 0, 'Atr-24': 0, 'Atr-25': 0, 'Atr-26': 0, 'Atr-27': 0, 'Atr-28': 0, 'Atr-29': 0, 'Atr-30': 0, 'Atr-31': 0, 'Atr-32': 0, 'Atr-33': 0, 'Atr-34': 0, 'Atr-35': 0, 'Atr-36': 0, 'Atr-37': 0, 'Atr-38': 0, 'Atr-39': 0, 'Atr-40': 0, 'Class': 0}；完全重复行：1235；重复特征行：1235
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## poker-8_vs_6.dat
- 解析：success；尺寸：47078 字节
- 维度：[1477, 11]；观察到末列：Class；类别：{'negative': 1460, 'positive': 17}
- 非数值列：['Class']；缺失：{'S1': 0, 'C1': 0, 'S2': 0, 'C2': 0, 'S3': 0, 'C3': 0, 'S4': 0, 'C4': 0, 'S5': 0, 'C5': 0, 'Class': 0}；完全重复行：2；重复特征行：2
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## winequality-white-3_vs_7.dat
- 解析：success；尺寸：60847 字节
- 维度：[900, 12]；观察到末列：Class；类别：{'negative': 880, 'positive': 20}
- 非数值列：['Class']；缺失：{'FixedAcidity': 0, 'VolatileAcidity': 0, 'CitricAcid': 0, 'ResidualSugar': 0, 'Chlorides': 0, 'FreeSulfurDioxide': 0, 'TotalSulfurDioxide': 0, 'Density': 0, 'PH': 0, 'Sulphates': 0, 'Alcohol': 0, 'Class': 0}；完全重复行：191；重复特征行：191
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## winequality-white-9_vs_4.dat
- 解析：success；尺寸：11767 字节
- 维度：[168, 12]；观察到末列：Class；类别：{'negative': 163, 'positive': 5}
- 非数值列：['Class']；缺失：{'FixedAcidity': 0, 'VolatileAcidity': 0, 'CitricAcid': 0, 'ResidualSugar': 0, 'Chlorides': 0, 'FreeSulfurDioxide': 0, 'TotalSulfurDioxide': 0, 'Density': 0, 'PH': 0, 'Sulphates': 0, 'Alcohol': 0, 'Class': 0}；完全重复行：10；重复特征行：10
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## yeast-1-2-8-9_vs_7.dat
- 解析：success；尺寸：44373 字节
- 维度：[947, 9]；观察到末列：Class；类别：{'negative': 917, 'positive': 30}
- 非数值列：['Class']；缺失：{'Mcg': 0, 'Gvh': 0, 'Alm': 0, 'Mit': 0, 'Erl': 0, 'Pox': 0, 'Vac': 0, 'Nuc': 0, 'Class': 0}；完全重复行：29；重复特征行：29
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## yeast5.dat
- 解析：success；尺寸：84965 字节
- 维度：[1484, 9]；观察到末列：Class；类别：{'negative': 1440, 'positive': 44}
- 非数值列：['Class']；缺失：{'Mcg': 0, 'Gvh': 0, 'Alm': 0, 'Mit': 0, 'Erl': 0, 'Pox': 0, 'Vac': 0, 'Nuc': 0, 'Class': 0}；完全重复行：31；重复特征行：31
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## yeast6.dat
- 解析：success；尺寸：84965 字节
- 维度：[1484, 9]；观察到末列：Class；类别：{'negative': 1449, 'positive': 35}
- 非数值列：['Class']；缺失：{'Mcg': 0, 'Gvh': 0, 'Alm': 0, 'Mit': 0, 'Erl': 0, 'Pox': 0, 'Vac': 0, 'Nuc': 0, 'Class': 0}；完全重复行：31；重复特征行：31
- 风险：标签语义、标签派生字段、近重复跨拆分、时间/组依赖和标识符不能仅凭静态文件排除；需人工核验。

## PDF 格式观察
- {'path': 'C:\\Users\\k64183565\\Desktop\\w\\xin\\郭论文.pdf', 'exists': True, 'fingerprint': {'path': 'C:\\Users\\k64183565\\Desktop\\w\\xin\\郭论文.pdf', 'exists': True, 'kind': 'file', 'bytes': 1891239, 'mtime_ns': 1789609499000000000, 'sha256': '64e77b0213ab7b1c6c6ff1febf26dd5c24feb1ce6a3db1d279462fa01951e60a'}, 'status': 'not_extracted', 'reason': "ModuleNotFoundError: No module named 'pypdf'", 'suggested_package': 'pypdf==4.3.1'}

import os
import cv2
import json
import pandas as pd



def get_pid_list(df, id):
    tmp_df = df[df['id'] == id]

    if tmp_df.empty or tmp_df['pid'].values[0] is None:
        return pd.DataFrame(columns=['id', 'pid'])
    else:
        return get_pid_list(df, tmp_df['pid'].values[0]).append(tmp_df)



f = open("./data/category.txt")
lines = f.readlines()

id_list = []
pid_list = []
permission_source = []
for line in lines:
    per_line = line.strip().split('    ')
    id_list.append( per_line[0])
    if per_line[1] != 'null':
        pid_list.append( per_line[1] )
    else:
        pid_list.append( None )

df = pd.DataFrame(
    {
        'id': id_list,
        'pid': pid_list
    }
)

id_list = df['id'].values
for i in id_list:

    pid_list = get_pid_list(df, i)['pid'].values.tolist()
    print(i, pid_list)
   


'''
13314 ['10019', '10168', '10885', '12204', '13092']
11270 ['10019', '10168', '10885', '12209', '10296']
11284 ['10004', '10282']
11287 ['10004', '10282']
10273 ['10004']
10282 ['10004']
10294 ['10004']
10295 ['10004']
10296 ['10019', '10168', '10885', '12209']
11321 ['10004', '10299']
13369 ['10011', '10088']
10299 ['10004']
10300 ['10004']
13372 ['10019', '10168', '10885', '12204']
10301 ['10004']
10302 ['10004', '13464']
13374 ['10015', '13658']
13376 ['10015', '10125', '10672', '11950']
13377 ['10015', '10125', '10672', '11950']
10306 ['10004', '13464']
10307 ['10004']
13381 ['10007', '10878']
11339 ['10004', '10300']
13388 ['10013']
13405 ['10015']
13406 ['10019', '10168', '10885', '12204']
10338 ['10005', '10055']
12387 ['10012', '11093']
11368 ['10019', '10168', '10885']
13417 ['10013']
13418 ['10007']
13419 ['10007', '10878']
13422 ['10005']
13423 ['10004', '10299']
13425 ['10013']
13426 ['10108']
10355 ['10005', '10060']
13427 ['10016', '10141']
10356 ['10005', '10060']
13428 ['10015', '10113']
13432 ['10108', '10756']
10362 ['10006']
10364 ['10006']
10365 ['10006']
10366 ['10006', '13493']
13439 ['10020']
10368 ['10006']
10369 ['10006', '13493']
13459 ['10012', '10095', '11827']
11412 ['10005', '10047']
13464 ['10004']
13469 ['10002', '10231']
13470 ['10002', '10231', '10127']
10399 ['10009']
13472 ['10002', '10231']
13473 ['10002', '10231']
13474 ['10002', '10231']
13475 ['10002', '10231']
10404 ['10009']
13476 ['10002', '10231']
10405 ['10009']
10406 ['10009']
10407 ['10009']
13479 ['10002', '10231']
10410 ['10009']
10411 ['10009']
10412 ['10009']
10413 ['10009']
13489 ['10013', '13425']
10418 ['10009']
10419 ['10009']
13491 ['10004']
10421 ['10009', '10081']
13493 ['10006']
10422 ['10009', '10081']
10423 ['10009', '10081']
13496 ['10017', '10145']
10431 ['10010', '10082']
10432 ['10010', '10082']
10434 ['10010', '10082']
10435 ['10010', '10082']
10436 ['10010', '10082']
10437 ['10010', '10082']
13509 ['10006']
10438 ['10010', '10082']
10439 ['10010', '10082']
10440 ['10010', '10082']
13513 ['10016', '10141', '10763']
10444 ['10010', '10083']
10445 ['10010', '10083']
10446 ['10011', '10085']
12494 ['10011', '10087']
10449 ['10011', '10085']
10452 ['10011', '10085']
10453 ['10011', '10085']
13525 ['10013', '13417']
10454 ['10011', '10085', '10473']
10456 ['10011', '10085']
13530 ['10013']
10460 ['10011', '10085']
10464 ['10011', '10085']
13536 ['10013', '10105']
10465 ['10011', '10085']
10466 ['10011', '10085']
10467 ['10011', '10085']
11491 ['10007', '13418']
10468 ['10011', '10085']
10469 ['10011', '10085']
11493 ['10007', '13418']
11496 ['10007', '13418']
10473 ['10011', '10085']
10474 ['10011', '10085']
10475 ['10011', '10085']
10476 ['10011', '10085']
10479 ['10011', '10085']
10480 ['10011', '10085']
11504 ['10007', '13418']
10481 ['10011', '10085']
10483 ['10011', '10085']
10486 ['10011', '10085']
10487 ['10011', '10085']
11511 ['10007', '10878', '13419']
10488 ['10011', '10085']
10491 ['10011', '10086']
10493 ['10011', '10087']
10494 ['10011', '10087']
13566 ['10013']
13567 ['10013', '13566']
11527 ['10009', '10412']
10504 ['10011', '10090']
10505 ['10011', '10090']
10507 ['10011', '10091']
10508 ['10011', '10091']
10509 ['10011', '10091']
10511 ['10011', '10091']
13584 ['10015', '10125', '10672', '11950', '13377']
13585 ['10015', '10125', '10672', '11950', '13377']
13586 ['10015', '10125', '10672', '11950', '13377']
10515 ['10011', '10091']
13588 ['10015', '10125', '10672', '11950', '13377']
11541 ['10009', '10406']
13589 ['10015', '10125', '10672', '11950', '13377']
10518 ['10011', '10091']
13590 ['10015', '10125', '10672', '11950', '13377']
10519 ['10011', '10091']
13591 ['10015', '10125', '10672', '11950', '13377']
10520 ['10011', '10091']
13596 ['10015', '10132']
10531 ['10012', '10095']
13605 ['10014']
13606 ['10014', '13605']
10536 ['10012', '10099']
13608 ['10014', '13605']
13609 ['10014', '13605']
13610 ['10014', '13605']
10539 ['10012', '10102']
13611 ['10014', '13605']
13612 ['10014', '13605']
12590 ['10004', '10282', '11287']
11567 ['10009', '10412']
10544 ['10012', '10102']
10547 ['10012', '10102']
13620 ['10014', '13605']
10550 ['10012', '10102']
10557 ['10013', '10103']
13633 ['10014', '13605']
11589 ['10009']
10566 ['10013', '10103']
10570 ['10013', '10109']
11597 ['10009', '10081', '10423']
11598 ['10009', '10081', '10423']
13647 ['10108']
10578 ['10017']
11604 ['10010', '10082']
13655 ['10001']
10584 ['10108']
13656 ['10001', '13655']
13657 ['10001', '13655']
13658 ['10015']
10591 ['10013', '10109']
10598 ['10013']
10603 ['10013', '10111']
11629 ['10010', '10082', '10439']
13680 ['10015', '13405']
10609 ['10108', '10112']
10611 ['10108', '10112']
13686 ['10018', '10163']
10615 ['10015', '10113']
10616 ['10015', '10113']
11642 ['10010', '10083', '10444']
10619 ['10021', '10177']
11643 ['10010', '10083', '10444']
11644 ['10010', '10083', '10444']
11645 ['10010', '10083', '10444']
11647 ['10010', '10083', '10444']
11648 ['10010', '10083', '10444']
11649 ['10010', '10083', '10445']
11650 ['10010', '10083', '10445']
11651 ['10010', '10083', '10445']
10628 ['10014', '10117']
13710 ['10007', '13418']
13714 ['10001', '10024']
13716 ['10108', '10756']
10645 ['10014', '10121']
11669 ['10011', '10085']
10648 ['10014', '10121']
13722 ['10011', '10091', '10507']
10653 ['10014', '10121']
12701 ['10019', '10168', '10885', '11368']
13725 ['10007', '13418']
10666 ['10014', '10123']
10668 ['10014', '10123']
13740 ['10006', '13509']
10669 ['10014', '10123']
10671 ['10015', '10125']
11695 ['10011', '10085', '10465']
10672 ['10015', '10125']
10679 ['10015', '10128']
10683 ['10015', '10130']
11710 ['10011', '10085', '10473']
13760 ['10013', '10103', '10557']
10691 ['10015', '10132']
10692 ['10015', '10132']
10693 ['10015', '10132']
11719 ['10011', '10085', '10475']
13767 ['10015', '10137']
13768 ['10007', '13418']
11721 ['10011', '10085', '10475']
13778 ['10021', '10176']
13779 ['10021', '10176']
13780 ['10021', '10176']
12757 ['10007', '13418']
12766 ['10007']
12767 ['10007']
13794 ['10015', '10125', '10672']
13797 ['10004', '10282']
12777 ['10015', '13658']
13801 ['10108', '10112']
10746 ['10016']
12795 ['10009', '10412']
11772 ['10011', '10089']
11773 ['10011', '10089']
12797 ['10009', '10412']
10752 ['10016', '10141']
10754 ['10016', '10141']
10756 ['10108']
11780 ['10011', '10091', '10507']
10757 ['10016', '10141']
10758 ['10016', '10141']
10759 ['10016', '10141']
11783 ['10011', '10091', '10508']
10761 ['10016', '10141']
13834 ['10015', '10125', '10672', '11950', '13377']
10763 ['10016', '10141']
10769 ['10017', '10142']
11793 ['10011', '10091']
10770 ['10017', '10142']
10776 ['10017']
10778 ['10017', '10144']
12828 ['10009', '10081', '10423']
10782 ['10017', '10146']
10783 ['10017', '10146']
10784 ['10017', '10146']
11813 ['10012', '10093']
11814 ['10012', '10093']
13863 ['10015', '10125', '10672', '11950', '13377']
11816 ['10012', '10093']
11818 ['10012', '10093']
11819 ['10012', '10095']
11820 ['10012', '10095']
11824 ['10012', '10095']
11825 ['10012', '10095']
12849 ['10011', '10085', '10446']
11827 ['10012', '10095']
11828 ['10012', '10095']
11831 ['10012', '10095']
11838 ['10012', '10097']
11839 ['10012', '10097']
11840 ['10012', '10097', '11841']
12864 ['10011', '10085', '11669']
11841 ['10012', '10097']
12865 ['10011', '10085', '11669']
11842 ['10012', '10097']
12866 ['10011', '10085', '10460']
12867 ['10011', '10085', '10460']
10829 ['10018', '10160']
11870 ['10020']
10857 ['10018', '10163']
11883 ['10013']
13934 []
13935 ['13934']
13936 ['13934']
13937 ['13934']
13938 ['13934']
13939 ['13934']
13940 ['13934', '13935']
13941 ['13934', '13935']
13942 ['13934', '13935']
10872 ['10019', '10167']
10873 ['10019', '10167']
13945 ['10015', '10125', '10672']
10876 ['10019', '10167']
10877 ['10019', '10167']
11901 ['10014', '10116']
10878 ['10007']
12926 ['10011', '10089', '11772']
10880 ['10019', '10167']
10881 ['10019', '10167']
10883 ['10019', '10168']
10885 ['10019', '10168']
10886 ['10019', '10168']
10889 ['10015', '10128']
10890 ['10020', '10169']
10891 ['10020', '10169']
11917 ['10021', '10176']
10899 ['10021', '10171']
12950 ['10012', '10095', '11824']
12952 ['10012', '10095', '11824']
11929 ['10015', '10125', '10671']
12956 ['10012', '10097', '11841']
12957 ['10012', '10097', '11841']
12960 ['10012', '10097', '11841']
10913 ['10021', '10176']
11937 ['10015', '10125', '10671']
12961 ['10012', '10097', '11841']
11939 ['10015', '10125', '10671']
11940 ['10015', '10125', '10671']
11941 ['10015', '10125', '10671']
11942 ['10015', '10125', '10672']
11943 ['10015', '10125', '10672']
10920 ['10021', '10176']
11944 ['10015', '10125', '10672']
11945 ['10015', '10125', '10672']
11946 ['10015', '10125', '10672']
10923 ['10021', '10176']
11948 ['10015', '10125', '10672']
10925 ['10021', '10176']
11949 ['10015', '10125', '10672']
11950 ['10015', '10125', '10672']
11963 ['10015', '10130', '10683']
11964 ['10015', '10130', '10683']
11965 ['10015', '10130', '10683']
12997 ['10015', '10125', '10672', '11942']
11980 ['10015', '10130']
10963 ['10001', '10024', '10185']
11987 ['10015', '10131']
11005 ['10002', '10227']
11006 ['10002', '10227']
11008 ['10002', '10227']
11009 ['10002', '10227']
11020 ['10002', '10030', '10235', '10052']
11022 ['10002', '10030', '10235']
10001 []
10002 []
10003 []
10004 []
11028 ['10002', '10030', '10236']
10005 []
10006 []
10007 []
10009 []
10010 []
10011 []
10012 []
10013 []
10014 []
10015 []
10016 []
10017 []
10018 []
12066 ['10017', '10142', '10769']
10019 []
12067 ['10017', '10142', '10769']
10020 []
13092 ['10019', '10168', '10885', '12204']
10021 []
13094 ['10007', '13418']
10024 ['10001']
10026 ['10001']
10027 ['10001']
10030 ['10002']
10031 ['10002']
10032 ['10003', '10042']
10035 ['10003']
12083 ['10017', '10146', '10783']
12084 ['10017', '10146', '10783']
10039 ['10003']
10040 ['10003']
10042 ['10003']
13116 ['10021', '10176', '12240']
10047 ['10005']
10048 ['10005']
10049 ['10005']
10050 ['10005']
10052 ['10002', '10030', '10235']
10054 ['10005']
10055 ['10005']
10057 ['10005']
10058 ['10005']
10059 ['10005']
10060 ['10005']
10061 ['10005', '10047']
12110 ['10004', '10295']
12113 ['10004', '10295']
11093 ['10012']
11099 ['10004']
10080 ['10010']
10081 ['10009']
10082 ['10010']
10083 ['10010']
10084 ['10010']
10085 ['10011']
10086 ['10011']
10087 ['10011']
10088 ['10011']
10089 ['10011']
10090 ['10011']
12138 ['10019', '10167', '10873']
10091 ['10011']
10092 ['10012']
10093 ['10012']
10094 ['10012']
10095 ['10012']
12143 ['10019', '10167', '10873']
10096 ['10012']
10097 ['10012']
10098 ['10012']
10099 ['10012']
10102 ['10012']
10103 ['10013']
10105 ['10013']
10106 ['10013']
10108 []
10109 ['10013']
10111 ['10013']
10112 ['10108']
10113 ['10015']
12161 ['10007', '10878', '13381']
10115 ['10014']
10116 ['10014']
10117 ['10014']
11141 ['10004', '10282']
10118 ['10011', '10089', '11772']
10120 ['10014', '10121']
10121 ['10014']
10123 ['10014']
10125 ['10015']
10127 ['10002', '10231']
10128 ['10015']
10130 ['10015']
10131 ['10015']
10132 ['10015']
10133 ['10015']
10135 ['10015']
12184 ['10019', '10168', '10883']
10137 ['10015']
12187 ['10019', '10168', '10883']
10141 ['10016']
10142 ['10017']
10144 ['10017']
10145 ['10017']
10146 ['10017']
10147 ['10017']
10150 ['10017']
10151 ['10017']
10153 ['10017']
10154 ['10017']
12204 ['10019', '10168', '10885']
12206 ['10019', '10168', '10885']
10159 ['10018']
10160 ['10018']
12209 ['10019', '10168', '10885']
10163 ['10018']
10167 ['10019']
10168 ['10019']
10169 ['10020']
10171 ['10021']
10176 ['10021']
10177 ['10021']
13250 ['10009', '10406']
12228 ['10020', '10169']
10182 ['10001', '10024']
10183 ['10001', '10024']
10185 ['10001', '10024']
13260 ['10009', '10413']
10191 ['10001']
12240 ['10021', '10176']
10193 ['10001']
10206 ['10001', '10026']
10209 ['10001', '10026']
13283 ['10012', '10097', '11841', '12956']
13284 ['10012', '10097', '11841', '12957']
12261 ['10021']
13286 ['10012', '10097', '11841', '12961']
12263 ['10021']
13289 ['10012', '10097', '11841', '12960']
13291 ['10012', '10097', '11841', '12960']
13292 ['10012', '10097', '11841', '12960']
13293 ['10012', '10097', '11841', '12960']
10224 ['10002']
13296 ['10012', '10097', '11841', '12960'pid_list']
13300 ['10012', '10097', '11841', '12961']
13301 ['10012', '10097', '11841', '12961']
10231 ['10002']
10235 ['10002', '10030']
10236 ['10002', '10030']
10238 ['10002', '10031']
'''

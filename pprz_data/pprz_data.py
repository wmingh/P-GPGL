#!/usr/bin/env python3

from __future__ import print_function, division  # 导入Python 2兼容特性，确保print函数和除法行为
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas库，用于数据处理和分析
from scipy.interpolate import griddata, interp1d  # 从scipy导入插值函数
from pprz_data import pprz_message_definitions as msg  # 导入Paparazzi消息定义模块
# import . import pprz_message_definitions as msg  # 相对导入（已注释）
import pdb  # 导入Python调试器
from scipy.interpolate import griddata, interp1d, CubicSpline

class DATA:
    """
    Data class from Paparazzi System.  # Paparazzi系统的数据类
    """

    def __init__(self, filename=None, ac_id=None, data_type=None, pad=10, sample_period=0.01):
        self.df_list = []  # 初始化数据帧列表
        self.filename = filename  # 数据文件名
        self.ac_id = ac_id  # 飞机ID
        self.df = None  # 主数据帧
        self.data_values = 0.  # 数据值
        self.data_type = data_type  # 数据类型
        self.pad = pad  # 时间填充值
        self.sample_period = sample_period  # 采样周期
        if self.data_type == 'fault':  # 如果是故障数据类型
            self.read_msg1_bundle()  # 读取第一组消息
        elif self.data_type == 'flight':  # 如果是飞行数据类型
            self.read_msg1_bundle()  # 读取第一组消息
            self.read_msg2_bundle()  # 读取第二组消息
            self.read_msg3_bundle()  # 读取第三组消息
        elif self.data_type == 'robust':  # 如果是鲁棒数据类型
            self.read_msg1_bundle()  # 读取第一组消息
            self.read_msg2_bundle()  # 读取第二组消息
            self.read_msg3_bundle()  # 读取第三组消息
            self.read_msg4_bundle()  # 读取第四组消息
        elif self.data_type == 'replay':  # 如果是重放数据类型
            self.read_replay_msg_bundle()  # 读取重放消息组

        self.find_min_max()  # 查找时间范围
        self.df_All = self.combine_dataframes()  # 合并所有数据帧

    def read_msg1_bundle(self):
        try:
            msg_name = 'attitude';
            columns = ['time', 'phi', 'psi', 'theta'];
            drop_columns = ['time']  # 姿态消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取姿态消息
        except:
            print(' Attitude msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'mode';
            columns = ['time', 'mode', '1', '2', '3', '4', '5'];
            drop_columns = ['time', '1', '2', '3', '4', '5']  # 模式消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取模式消息
        except:
            print('Paparazzi Mode msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imuaccel';
            columns = ['time', 'Ax', 'Ay', 'Az'];
            drop_columns = ['time']  # IMU加速度消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取IMU加速度消息
        except:
            print(' IMU Acceleration msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imuaccel_scaled';
            columns = ['time', 'Ax_sca', 'Ay_sca', 'Az_sca'];
            drop_columns = ['time']  # 缩放IMU加速度消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取缩放IMU加速度消息
        except:
            print(' IMU Scaled Acceleration msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imuaccel_raw';
            columns = ['time', 'Ax_raw', 'Ay_raw', 'Az_raw'];
            drop_columns = ['time']  # 原始IMU加速度消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取原始IMU加速度消息
        except:
            print(' IMU Raw Acceleration msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'gps';
            columns = ['time', '1', 'east', 'north', 'course', 'alt', 'vel', 'climb', '8', '9', '10', '11'];
            drop_columns = ['time', '1', '8', '9', '10', '11']  # GPS消息定义
            df = self.extract_message(msg_name, columns, drop_columns)  # 提取GPS消息
            df.alt = df.alt / 1000.  # 转换高度单位
            df.vel = df.vel / 100.  # 转换速度单位为m/s
            df.climb = df.climb / 100.  # 转换爬升率为m/s
            print(' Generating 3D velocity...')  # 打印信息
            df['vel_3d'] = df.climb.apply(lambda x: x ** 2)  # 计算垂直速度平方
            df.vel_3d = df.vel_3d + df.vel.apply(lambda x: x ** 2)  # 加上水平速度平方
            df.vel_3d = df.vel_3d.apply(lambda x: np.sqrt(x))  # 计算3D速度模长
            #             if 1:
            #                 # Calculate 3D speed (including the vertical component to the horizontal speed on ground.)
            #                 print(' Calculating the 3D speed norm !')
            #                 df['vel_3d1'] = df.climb.apply(lambda x: x**2)
            #                 print(df.vel_3d1.any())
            self.df_list.append(df)  # 添加处理后的GPS数据
        except:
            print(' GPS msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imugyro';
            columns = ['time', 'Gx', 'Gy', 'Gz'];
            drop_columns = ['time']  # IMU陀螺仪消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取IMU陀螺仪消息
        except:
            print(' IMU Gyro msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imugyro_scaled';
            columns = ['time', 'Gx_sca', 'Gy_sca', 'Gz_sca'];
            drop_columns = ['time']  # 缩放IMU陀螺仪消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取缩放IMU陀螺仪消息
        except:
            print(' IMU Scaled Gyro msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'imugyro_raw';
            columns = ['time', 'Gx_raw', 'Gy_raw', 'Gz_raw'];
            drop_columns = ['time']  # 原始IMU陀螺仪消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取原始IMU陀螺仪消息
        except:
            print(' IMU Raw Gyro msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'fault_telemetry';
            columns = ['time', 'Fault_Telemetry'];
            drop_columns = ['time']  # 故障遥测消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取故障遥测消息
        except:
            print(' Fault Telemetry msg doesnt exist ')  # 异常处理

    def read_msg2_bundle(self):
        try:
            msg_name = 'actuators';
            columns = ['time', 'S0', 'S1', 'S2'];
            drop_columns = ['time']  # 执行器消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取执行器消息
        except:
            print(' Actuators msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'commands';
            columns = ['time', 'C0', 'C1', 'C2'];
            drop_columns = ['time']  # 命令消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取命令消息
        except:
            print(' Commands msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'energy_new';
            columns = ['time', 'Throttle', 'Volt', 'Amp', 'Watt', 'mAh', 'Wh'];
            drop_columns = ['time']  # 能量消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取能量消息
        except:
            print(' Energy_new msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'air_data';
            columns = ['time', 'Ps', 'Pdyn_AD', 'temp', 'qnh', 'amsl_baro', 'airspeed', 'TAS'];
            drop_columns = ['time']  # 大气数据消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取大气数据消息
        except:
            print(' Air Data msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'desired';
            columns = ['time', 'D_roll', 'D_pitch', 'D_course', 'D_x', 'D_y', 'D_altitude', 'D_climb', 'D_airspeed'];
            drop_columns = ['time']  # 期望值消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取期望值消息
        except:
            print(' Desired msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'actuators_4';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz'];
            drop_columns = ['time']  # 4值执行器消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取4值执行器消息
        except:
            print(' 4-valued Actuators msg doesnt exist ')  # 异常处理

    def read_msg3_bundle(self):
        try:
            msg_name = 'gust';
            columns = ['time', 'wx', 'wz', 'Va_gust', 'gamma_gust', ' AoA_gust', 'theta_com_gust'];
            drop_columns = ['time']  # 阵风消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取阵风消息
        except:
            print(' Gust msg does not exist ')  # 异常处理
        # <message name="SOARING_TELEMETRY" id="212">
        # <field name="velocity"     type="float"  unit="m/s">veocity</field>
        # <field name="a_attack"     type="float"  unit="rad">angle of attack</field>
        # <field name="a_sideslip"   type="float"  unit="rad">sideslip angle</field>
        # <field name="dynamic_p"    type="float"  unit="Pa"/>
        # <field name="static_p"     type="float"  unit="Pa"/>
        # <field name="wind_x"       type="float"  unit="m/s"/>
        # <field name="wind_z"       type="float"  unit="m/s"/>
        # <field name="wind_x_dot"   type="float"  unit="m/s2"/>
        # <field name="wind_z_dot"   type="float"  unit="m/s2"/>
        # <field name="wind_power"   type="float"  unit="W"/>
        try:
            msg_name = 'soaring_telemetry';
            columns = ['time', 'sp_Va', 'sp_aoa', 'sp_beta', 'sp_dyn_p', 'sp_sta_p', 'sp_wx', 'sp_wz', 'sp_d_wx',
                       'sp_d_wz', 'sp_w_power'];
            drop_columns = ['time']  # 滑翔遥测消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取滑翔遥测消息
        except:
            print(' Soaring Telemetry msg does not exist ')  # 异常处理
        # <message name="ROTORCRAFT_FP" id="147">
        #   <field name="east"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="north"    type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="up"       type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="veast"    type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="vnorth"   type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="vup"      type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
        #   <field name="phi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="theta"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="psi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="carrot_east"   type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_north"  type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_up"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
        #   <field name="carrot_psi"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
        #   <field name="thrust"        type="int32"/>
        #   <field name="flight_time"   type="uint16" unit="s"/>
        # </message>
        try:
            msg_name = 'rotorcraft_fp';
            columns = ['time', 'east', 'north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
                       'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time'];
            drop_columns = ['time']  # 旋翼飞行器飞行点消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取旋翼飞行器飞行点消息
        except:
            print(' Rotorcraft_fp msg does not exist ')  # 异常处理

    def read_msg4_bundle(self):
        try:
            '''This one is a bit hardcoded !!! Sorry ! '''  # 注释：这部分有点硬编码
            motor_df_list = msg.read_log_dshot_telemetry(self.ac_id, self.filename)  # 读取DSHOT遥测数据
            for df in motor_df_list:  # 遍历电机数据帧列表
                self.df_list.append(df)  # 添加到数据帧列表
        except:
            print(' DSHOT TELEMETRY msg does not exist ')  # 异常处理
        try:
            msg_name = 'payload6';
            columns = ['time', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'];
            drop_columns = ['time']  # 载荷6消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取载荷6消息
        except:
            print(' PAYLOAD6 msg does not exist ')  # 异常处理
        try:
            msg_name = 'actuators_4';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz'];
            drop_columns = ['time']  # 4值执行器消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取4值执行器消息
        except:
            print(' 4-valued Actuators msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'actuators_6';
            columns = ['time', 'M1_pprz', 'M2_pprz', 'M3_pprz', 'M4_pprz', 'M5_pprz', 'M6_pprz'];
            drop_columns = ['time']  # 6值执行器消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取6值执行器消息
        except:
            print(' 6-valued Actuators msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'actuators_8';
            columns = ['time', 'S1', 'S2', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'];
            drop_columns = ['time']  # 8值执行器消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取8值执行器消息
        except:
            print(' 8-valued Actuators msg doesnt exist ')  # 异常处理
        try:
            msg_name = 'rotorcraft_fault';
            columns = ['time', 'M1F', 'M2F', 'M3F', 'M4F', 'M5F', 'M6F'];
            drop_columns = ['time']  # 旋翼飞行器故障消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取旋翼飞行器故障消息
        except:
            print(' ROTORCRAFT FAULT msg does not exist (hexa-version)')  # 异常处理
        try:
            msg_name = 'adc_consumptions';
            columns = ['time', 'Pow1', 'Pow2'];
            drop_columns = ['time']  # ADC消耗消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取ADC消耗消息
        except:
            print(' ADC_CONSUMPTIONS msg does not exist ')  # 异常处理
        try:
            msg_name = 'robust_morph_angle';
            columns = ['time', 'Morph1', 'Morph2'];
            drop_columns = ['time']  # 鲁棒变形角度消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取鲁棒变形角度消息
        except:
            print(' MORPH_ANGLE msg does not exist (This is for RoBust-Morphing-Hexa)')  # 异常处理

    def read_replay_msg_bundle(self):
        try:
            msg_name = 'rotorcraft_fp';
            columns = ['time', 'east', 'north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
                       'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time'];
            drop_columns = ['time']  # 旋翼飞行器飞行点消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取旋翼飞行器飞行点消息
        except:
            print(' Rotorcraft_fp msg does not exist ')  # 异常处理
        try:
            msg_name = 'robust_morph_angle';
            columns = ['time', 'Morph1', 'Morph2'];
            drop_columns = ['time']  # 鲁棒变形角度消息定义
            self.df_list.append(self.extract_message(msg_name, columns, drop_columns))  # 提取鲁棒变形角度消息
        except:
            pr  # 不完整的异常处理（代码错误）

    def get_settings(self):
        ''' Special Message used for the fault injection settings
        2 multiplicative, and 2 additive, and only appears when we change them
        so the time between has to be filled in...'''  # 用于故障注入设置的特殊消息
        msg_name = 'settings';
        columns = ['time', 'm1', 'm2', 'add1', 'add2'];
        drop_columns = ['time']  # 设置消息定义
        df = self.extract_message(msg_name, columns, drop_columns)  # 提取设置消息
        df.add1 = df.add1 / 9600.;
        df.add2 = df.add2 / 9600.  # 转换添加值
        return df  # 返回设置数据帧

    def extract_message(self, msg_name, columns, drop_columns):
        ''' Given msg names such as attitute, we will call msg.read_log_attitute'''  # 根据消息名称调用相应的读取函数
        exec('self.data_values = msg.read_log_{}(self.ac_id, self.filename)'.format(msg_name))  # 动态执行消息读取函数
        df = pd.DataFrame(self.data_values, columns=columns)  # 创建数据帧
        df.index = df.time  # 设置时间索引
        df.drop(drop_columns, axis=1, inplace=True)  # 删除指定列
        return df  # 返回处理后的数据帧

    def find_min_max(self):
        self.min_t = 1000.  # 初始化最小时间
        self.max_t = -1.  # 初始化最大时间
        for df in self.df_list:  # 遍历所有数据帧
            self.min_t = min(self.min_t, min(df.index))  # 更新最小时间
            self.max_t = max(self.max_t, max(df.index))  # 更新最大时间
        print('Min time :', self.min_t, 'Maximum time :', self.max_t)  # 打印时间范围 # 最小时间可能有误导...可能需要更好的方法

#三次样条插值
    # def linearize_time(self, df, min_t=None, max_t=None):
    #     """
    #     使用三次样条插值进行时间线性化
    #     """
    #     if (min_t or max_t) == None:  # 如果未提供时间范围
    #         min_t = min(df.index)  # 使用数据帧的最小时间
    #         max_t = max(df.index)  # 使用数据帧的最大时间
    #
    #     time = np.arange(int(min_t) + self.pad, int(max_t) - self.pad, self.sample_period)  # 生成线性时间序列
    #     out = pd.DataFrame()  # 创建输出数据帧
    #     out['time'] = time  # 添加时间列
    #
    #     for col in df.columns:  # 遍历所有列
    #         # 检查数据点数量是否足够进行三次样条插值（至少需要4个点）
    #         if len(df.index) >= 4:
    #             try:
    #                 # 使用三次样条插值
    #                 cs = CubicSpline(df.index, df[col])
    #                 out[col] = cs(time)
    #             except Exception as e:
    #                 print(f"警告: 对列 {col} 使用三次样条插值失败，回退到线性插值。错误: {e}")
    #                 # 回退到线性插值
    #                 func = interp1d(df.index, df[col], fill_value='extrapolate')
    #                 out[col] = func(time)
    #         else:
    #             # 数据点不足，使用线性插值
    #             print(f"警告: 列 {col} 数据点不足 ({len(df.index)}个)，使用线性插值代替三次样条")
    #             func = interp1d(df.index, df[col], fill_value='extrapolate')
    #             out[col] = func(time)
    #
    #     out.index = out.time  # 设置时间索引
    #     out.drop(['time'], axis=1, inplace=True)  # 删除时间列
    #     return out  # 返回线性化后的数据帧
#线性插值
    def linearize_time(self, df, min_t=None, max_t=None):
        if (min_t or max_t) == None:  # 如果未提供时间范围
            min_t = min(df.index)  # 使用数据帧的最小时间
            max_t = max(df.index)  # 使用数据帧的最大时间
        time = np.arange(int(min_t) + self.pad, int(max_t) - self.pad, self.sample_period)  # 生成线性时间序列
        out = pd.DataFrame()  # 创建输出数据帧
        out['time'] = time  # 添加时间列
        for col in df.columns:  # 遍历所有列
            func = interp1d(df.index, df[col], fill_value='extrapolate')  # 创建插值函数 # FIXME : 如果想使用线性插值之外的方法
            out[col] = func(time)  # 应用插值
        out.index = out.time  # 设置时间索引
        out.drop(['time'], axis=1, inplace=True)  # 删除时间列
        return out  # 返回线性化后的数据帧

    def combine_dataframes(self):
        frames = [self.linearize_time(df, self.min_t, self.max_t) for df in self.df_list]  # 线性化所有数据帧
        return pd.concat(frames, axis=1, ignore_index=False, sort=False)  # 合并所有数据帧

    def combine_settings_dataframe(self):
        df_settings = self.get_settings()  # 获取设置数据
        df = self.df_All.copy()  # 复制主数据帧
        # df["m1"] = np.NaN ; df["m2"] = np.NaN ; df["add1"] = np.NaN ; df["add2"] = np.NaN  # 初始化设置列（已注释）
        df["m1"] = np.nan;  # 初始化m1列为NaN
        df["m2"] = np.nan;  # 初始化m2列为NaN
        df["add1"] = np.nan;  # 初始化add1列为NaN
        df["add2"] = np.nan  # 初始化add2列为NaN
        # Starting time  # 起始时间
        i_st = df.index[0]  # 获取起始索引
        for i in range(len(df_settings.index)):  # 遍历设置数据索引
            row_pos = int(round(df_settings.index[i] - i_st, 1) / self.sample_period)
            row_label = df.index[row_pos]
            df.at[row_label, "m1"] = df_settings["m1"].iloc[i]  # 设置m1值
            df.at[row_label, "m2"] = df_settings["m2"].iloc[i]  # 设置m2值
            df.at[row_label, "add1"] = df_settings["add1"].iloc[i]  # 设置add1值
            df.at[row_label, "add2"] = df_settings["add2"].iloc[i]  # 设置add2值
        return df  # 返回合并后的数据帧

    def get_labelled_data(self):
        df = self.combine_settings_dataframe()  # 获取合并设置的数据帧
        first_row = df.index[0]
        df.at[first_row, "m1"] = 1.0  # 设置初始m1值
        df.at[first_row, "m2"] = 1.0  # 设置初始m2值
        df.at[first_row, "add1"] = 0.0  # 设置初始add1值
        df.at[first_row, "add2"] = 0.0  # 设置初始add2值
        return df.ffill()  # 前向填充NaN值并返回

    # def combine_settings_dataframe(self):
    #     df_settings = self.get_settings() #FIXME : we may check if this has been already done before or not...  # 获取设置数据
    #     return pd.concat(([self.df_All, df_settings]), axis=1, ignore_index=False, sort=False)  # 合并数据帧

    # def get_labelled_data(self):
    #     df = self.combine_settings_dataframe()  # 获取合并设置的数据帧
    #     df.m1.iloc[0] = 1.0  # 设置初始m1值
    #     df.m2.iloc[0] = 1.0  # 设置初始m2值
    #     df.add1.iloc[0] = 0.0  # 设置初始add1值
    #     df.add2.iloc[0] = 0.0  # 设置初始add2值
    #     return df.ffill()  # 前向填充NaN值并返回
#11111111111111
# #!/usr/bin/env python3
#
# from __future__ import print_function, division
# import numpy as np
# import pandas as pd
# from scipy.interpolate import griddata, interp1d
# from pprz_data import pprz_message_definitions as msg
# #from . import pprz_message_definitions as msg
# import pdb
#
# class DATA:
#     """
#     Data class from Paparazzi System.
#     """
#     def __init__(self, filename=None, ac_id=None, data_type=None, pad=10, sample_period=0.01):
#         self.df_list = []
#         self.filename = filename
#         self.ac_id = ac_id
#         self.df = None
#         self.data_values = 0.
#         self.data_type = data_type
#         self.pad = pad
#         self.sample_period = sample_period
#         if self.data_type=='fault':
#             self.read_msg1_bundle()
#         elif self.data_type=='flight':
#             self.read_msg1_bundle()
#             self.read_msg2_bundle()
#             self.read_msg3_bundle()
#         elif self.data_type=='robust':
#             self.read_msg1_bundle()
#             self.read_msg2_bundle()
#             self.read_msg3_bundle()
#             self.read_msg4_bundle()
#         elif self.data_type=='replay':
#             self.read_replay_msg_bundle()
#
#         self.find_min_max()
#         self.df_All = self.combine_dataframes()
#
#
#     def read_msg1_bundle(self):
#         try:
#             msg_name = 'attitude' ;columns=['time', 'phi','psi','theta'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Attitude msg doesnt exist ')
#         try:
#             msg_name= 'mode'; columns=['time','mode','1','2','3','4','5']; drop_columns = ['time','1','2','3','4','5']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print('Paparazzi Mode msg doesnt exist ')
#         try:
#             msg_name = 'imuaccel';columns=['time','Ax','Ay','Az']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Acceleration msg doesnt exist ')
#         try:
#             msg_name = 'imuaccel_scaled';columns=['time','Ax_sca','Ay_sca','Az_sca']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Scaled Acceleration msg doesnt exist ')
#         try:
#             msg_name = 'imuaccel_raw';columns=['time','Ax_raw','Ay_raw','Az_raw']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Raw Acceleration msg doesnt exist ')
#         try:
#             msg_name='gps';columns=['time','1','east','north','course','alt', 'vel', 'climb', '8','9','10','11'];drop_columns=['time','1','8','9','10','11']
#             df = self.extract_message( msg_name, columns, drop_columns)
#             df.alt = df.alt/1000.
#             df.vel = df.vel/100.     #convert to m/s
#             df.climb = df.climb/100. #convert to m/s
#             print(' Generating 3D velocity...')
#             df['vel_3d'] = df.climb.apply(lambda x: x**2)
#             df.vel_3d = df.vel_3d + df.vel.apply(lambda x: x**2)
#             df.vel_3d = df.vel_3d.apply(lambda x: np.sqrt(x))
# #             if 1:
# #                 # Calculate 3D speed (including the vertical component to the horizontal speed on ground.)
# #                 print(' Calculating the 3D speed norm !')
# #                 df['vel_3d1'] = df.climb.apply(lambda x: x**2)
# #                 print(df.vel_3d1.any())
#             self.df_list.append(df)
#         except: print(' GPS msg doesnt exist ')
#         try:
#             msg_name = 'imugyro';columns=['time','Gx','Gy','Gz']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Gyro msg doesnt exist ')
#         try:
#             msg_name = 'imugyro_scaled';columns=['time','Gx_sca','Gy_sca','Gz_sca']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Scaled Gyro msg doesnt exist ')
#         try:
#             msg_name = 'imugyro_raw';columns=['time','Gx_raw','Gy_raw','Gz_raw']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' IMU Raw Gyro msg doesnt exist ')
#         try:
#             msg_name = 'fault_telemetry';columns=['time','Fault_Telemetry']; drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Fault Telemetry msg doesnt exist ')
#
#     def read_msg2_bundle(self):
#         try:
#             msg_name = 'actuators' ;columns=['time', 'S0','S1','S2'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Actuators msg doesnt exist ')
#         try:
#             msg_name = 'commands' ;columns=['time', 'C0','C1','C2'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Commands msg doesnt exist ')
#         try:
#             msg_name = 'energy_new' ;columns=['time', 'Throttle', 'Volt', 'Amp', 'Watt', 'mAh', 'Wh'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Energy_new msg doesnt exist ')
#         try:
#             msg_name = 'air_data' ;columns=['time', 'Ps', 'Pdyn_AD', 'temp', 'qnh', 'amsl_baro', 'airspeed', 'TAS'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Air Data msg doesnt exist ')
#         try:
#             msg_name = 'desired'; columns=['time','D_roll','D_pitch','D_course','D_x', 'D_y', 'D_altitude','D_climb','D_airspeed']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Desired msg doesnt exist ')
#         try:
#             msg_name = 'actuators_4' ;columns=['time','M1_pprz','M2_pprz','M3_pprz','M4_pprz'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' 4-valued Actuators msg doesnt exist ')
#
#     def read_msg3_bundle(self):
#         try:
#             msg_name = 'gust' ; columns=['time','wx','wz', 'Va_gust', 'gamma_gust', ' AoA_gust', 'theta_com_gust']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Gust msg does not exist ')
#         # <message name="SOARING_TELEMETRY" id="212">
# # <field name="velocity"     type="float"  unit="m/s">veocity</field>
# # <field name="a_attack"     type="float"  unit="rad">angle of attack</field>
# # <field name="a_sideslip"   type="float"  unit="rad">sideslip angle</field>
# # <field name="dynamic_p"    type="float"  unit="Pa"/>
# # <field name="static_p"     type="float"  unit="Pa"/>
# # <field name="wind_x"       type="float"  unit="m/s"/>
# # <field name="wind_z"       type="float"  unit="m/s"/>
# # <field name="wind_x_dot"   type="float"  unit="m/s2"/>
# # <field name="wind_z_dot"   type="float"  unit="m/s2"/>
# # <field name="wind_power"   type="float"  unit="W"/>
#         try:
#             msg_name = 'soaring_telemetry' ; columns=['time','sp_Va','sp_aoa','sp_beta','sp_dyn_p','sp_sta_p','sp_wx','sp_wz','sp_d_wx','sp_d_wz','sp_w_power']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Soaring Telemetry msg does not exist ')
#   # <message name="ROTORCRAFT_FP" id="147">
#   #   <field name="east"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="north"    type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="up"       type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="veast"    type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
#   #   <field name="vnorth"   type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
#   #   <field name="vup"      type="int32" alt_unit="m/s" alt_unit_coef="0.0000019"/>
#   #   <field name="phi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
#   #   <field name="theta"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
#   #   <field name="psi"      type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
#   #   <field name="carrot_east"   type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="carrot_north"  type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="carrot_up"     type="int32" alt_unit="m" alt_unit_coef="0.0039063"/>
#   #   <field name="carrot_psi"    type="int32" alt_unit="deg" alt_unit_coef="0.0139882"/>
#   #   <field name="thrust"        type="int32"/>
#   #   <field name="flight_time"   type="uint16" unit="s"/>
#   # </message>
#         try:
#             msg_name = 'rotorcraft_fp' ; columns=['time','east','north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
#                                                     'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Rotorcraft_fp msg does not exist ')
#
#     def read_msg4_bundle(self):
#         try:
#             '''This one is a bit hardcoded !!! Sorry ! '''
#             motor_df_list = msg.read_log_dshot_telemetry(self.ac_id, self.filename)
#             for df in motor_df_list:
#                 self.df_list.append(df)
#         except: print(' DSHOT TELEMETRY msg does not exist ')
#         try:
#             msg_name = 'payload6' ; columns=['time','M1','M2','M3','M4','M5','M6']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' PAYLOAD6 msg does not exist ')
#         try:
#             msg_name = 'actuators_4' ;columns=['time','M1_pprz','M2_pprz','M3_pprz','M4_pprz'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' 4-valued Actuators msg doesnt exist ')
#         try:
#             msg_name = 'actuators_6' ;columns=['time','M1_pprz','M2_pprz','M3_pprz','M4_pprz','M5_pprz','M6_pprz'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' 6-valued Actuators msg doesnt exist ')
#         try:
#             msg_name = 'actuators_8' ;columns=['time','S1','S2','M1','M2','M3','M4','M5','M6'] ;drop_columns = ['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' 8-valued Actuators msg doesnt exist ')
#         try:
#             msg_name = 'rotorcraft_fault' ; columns=['time','M1F','M2F','M3F','M4F','M5F','M6F']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' ROTORCRAFT FAULT msg does not exist (hexa-version)')
#         try:
#             msg_name = 'adc_consumptions' ; columns=['time','Pow1','Pow2']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' ADC_CONSUMPTIONS msg does not exist ')
#         try:
#             msg_name = 'robust_morph_angle' ; columns=['time','Morph1','Morph2']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' MORPH_ANGLE msg does not exist (This is for RoBust-Morphing-Hexa)')
#
#     def read_replay_msg_bundle(self):
#         try:
#             msg_name = 'rotorcraft_fp' ; columns=['time','east','north', 'up', 'veast', 'vnorth', 'vup', 'phi', 'theta', 'psi', 'carrot_east',
#                                                     'carrot_north', 'carrot_up', 'carrot_psi', 'thrust', 'flight_time']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: print(' Rotorcraft_fp msg does not exist ')
#         try:
#             msg_name = 'robust_morph_angle' ; columns=['time','Morph1','Morph2']; drop_columns=['time']
#             self.df_list.append( self.extract_message( msg_name, columns, drop_columns) )
#         except: pr
#
#     def get_settings(self):
#         ''' Special Message used for the fault injection settings
#         2 multiplicative, and 2 additive, and only appears when we change them
#         so the time between has to be filled in...'''
#         msg_name = 'settings'; columns=['time','m1','m2','add1','add2'];drop_columns=['time']
#         df = self.extract_message( msg_name, columns, drop_columns)
#         df.add1 = df.add1/9600. ; df.add2 = df.add2/9600.
#         return df
#
#     def extract_message(self, msg_name, columns, drop_columns):
#         ''' Given msg names such as attitute, we will call msg.read_log_attitute'''
#         exec('self.data_values = msg.read_log_{}(self.ac_id, self.filename)'.format(msg_name))
#         df = pd.DataFrame(self.data_values, columns=columns)
#         df.index = df.time
#         df.drop(drop_columns, axis=1, inplace=True)
#         return df
#
#     def find_min_max(self):
#         self.min_t = 1000.
#         self.max_t = -1.
#         for df in self.df_list:
#             self.min_t = min(self.min_t, min(df.index))
#             self.max_t = max(self.max_t, max(df.index))
#         print('Min time :',self.min_t,'Maximum time :', self.max_t) # Minimum time can be deceiving... we may need to find a better way.
#
#     def linearize_time(self, df, min_t=None, max_t=None):
#         if (min_t or max_t) == None:
#             min_t = min(df.index)
#             max_t = max(df.index)
#         time = np.arange(int(min_t)+self.pad, int(max_t)-self.pad, self.sample_period)
#         out = pd.DataFrame()
#         out['time'] = time
#         for col in df.columns:
#             func = interp1d(df.index , df[col], fill_value='extrapolate') # FIXME : If we want to use a different method other than linear interpolation.
#             out[col] = func(time)
#         out.index = out.time
#         out.drop(['time'], axis=1, inplace=True)
#         return out
#
#     def combine_dataframes(self):
#         frames = [self.linearize_time(df, self.min_t, self.max_t) for df in self.df_list]
#         return pd.concat(frames, axis=1, ignore_index=False, sort=False)
#
#     def combine_settings_dataframe(self):
#         df_settings = self.get_settings()
#         df = self.df_All.copy()
#         #df["m1"] = np.NaN ; df["m2"] = np.NaN ; df["add1"] = np.NaN ; df["add2"] = np.NaN
#         df["m1"] = np.nan;
#         df["m2"] = np.nan;
#         df["add1"] = np.nan;
#         df["add2"] = np.nan
#         # Starting time
#         i_st=df.index[0]
#         for i in range(len(df_settings.index)):
#             df.m1.iloc[int(round(df_settings.index[i]-i_st,1)/self.sample_period)] = df_settings.m1.iloc[i]
#             df.m2.iloc[int(round(df_settings.index[i]-i_st,1)/self.sample_period)] = df_settings.m2.iloc[i]
#             df.add1.iloc[int(round(df_settings.index[i]-i_st,1)/self.sample_period)] = df_settings.add1.iloc[i]
#             df.add2.iloc[int(round(df_settings.index[i]-i_st,1)/self.sample_period)] = df_settings.add2.iloc[i]
#         return df
#
#     def get_labelled_data(self):
#         df = self.combine_settings_dataframe()
#         df.m1.iloc[0] = 1.0
#         df.m2.iloc[0] = 1.0
#         df.add1.iloc[0] = 0.0
#         df.add2.iloc[0] = 0.0
#         return df.ffill()
#
#     # def combine_settings_dataframe(self):
#     #     df_settings = self.get_settings() #FIXME : we may check if this has been already done before or not...
#     #     return pd.concat(([self.df_All, df_settings]), axis=1, ignore_index=False, sort=False)
#
#     # def get_labelled_data(self):
#     #     df = self.combine_settings_dataframe()
#     #     df.m1.iloc[0] = 1.0
#     #     df.m2.iloc[0] = 1.0
#     #     df.add1.iloc[0] = 0.0
#     #     df.add2.iloc[0] = 0.0
#     #     return df.ffill()

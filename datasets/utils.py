
class Pinyin(object):
    """docstring for Pinyin"""
    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un', 'uo', 'v', 've']
        self.shengdiao= ['1', '2', '3', '4', '5']
        self.sm_size=len(self.shengmu)+1
        self.ym_size=len(self.yunmu)+1
        self.sd_size=len(self.shengdiao)+1

    def get_sm_ym_sd(self, pinyin):
        s=pinyin
        assert isinstance(s, str),'input of function get_sm_ym_sd is not string'
        assert s[-1] in '12345',f'input of function get_sm_ym_sd is not valid,{s}'
        sm, ym, sd = None, None, None
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == None:
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]
        return sm, ym, sd
    
    def get_sm_ym_sd_labels(self, pinyin):
        sm, ym, sd=self.get_sm_ym_sd(pinyin)
        return self.shengmu.index(sm)+1 if sm in self.shengmu else 0, \
            self.yunmu.index(ym)+1 if ym in self.yunmu else 0, \
                self.shengdiao.index(sd)+1 if sd in self.shengdiao else 0
    
    def get_pinyinstr(self, sm_ym_sd_label):
        sm, ym, sd = sm_ym_sd_label
        sm -= 1
        ym -= 1
        sd -= 1
        sm = self.shengmu[sm] if sm >=0 else ''
        ym = self.yunmu[ym] if ym >= 0 else ''
        sd = self.shengdiao[sd] if sd >= 0 else ''
        return sm + ym + sd

pho_convertor = Pinyin()

if __name__=='__main__':
    print(pho_convertor.get_sm_ym_sd_labels('a1'),type(pho_convertor.get_sm_ym_sd_labels('a1')))
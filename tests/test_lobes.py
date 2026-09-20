from ehrdoc.venn.lobes import encode_lobe

def test_encode_lobe():
    assert encode_lobe(True,True,True)==0
    assert encode_lobe(True,True,False)==1
    assert encode_lobe(True,False,True)==2
    assert encode_lobe(True,False,False)==3
    assert encode_lobe(False,True,True)==4
    assert encode_lobe(False,True,False)==5
    assert encode_lobe(False,False,True)==6
    assert encode_lobe(False,False,False)==-1

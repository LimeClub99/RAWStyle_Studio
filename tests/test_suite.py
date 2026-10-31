"""Offline tests. Run: python -m unittest discover -s tests -v"""
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest


class CatalogTests(unittest.TestCase):
    def test_parser_gets_top_level_kelvin_not_nested_look(self):
        from rawstyle.catalog import parse_develop_table
        text = '''s = { WhiteBalance = "As Shot", Temperature = 7779, Tint = -3,
            Look = { Parameters = { Temperature = 1000, Tint = 100 } },
            Exposure2012 = 1.16, Contrast2012 = 18, FilterList = {} }'''
        result = parse_develop_table(text)
        self.assertEqual(result['Temperature'], 7779)
        self.assertEqual(result['Tint'], -3)
        self.assertEqual(result['Exposure2012'], 1.16)
        self.assertNotIn('WhiteBalance', result)

    def test_read_only_catalog_pairs_files_and_skips_duplicates(self):
        from rawstyle.catalog import read_catalog, pair_raws
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'fixture.lrcat'
            con = sqlite3.connect(db)
            con.executescript('''
                create table Adobe_images (id_local integer, rootFile integer);
                create table AgLibraryFile (id_local integer, baseName text, extension text);
                create table Adobe_imageDevelopSettings (image integer, text text);
            ''')
            for j in range(20):
                con.execute('insert into Adobe_images values (?,?)', (j,j))
                con.execute('insert into AgLibraryFile values (?,?,?)',(j,f'TEST{j:03d}','CR2'))
                con.execute('insert into Adobe_imageDevelopSettings values (?,?)',
                            (j,f's = {{ WhiteBalance = "As Shot", Temperature = {3000+j*100}, Tint = {j-5}, Exposure2012 = 0.3 }}'))
            con.commit(); con.close()
            records = read_catalog(db)
            self.assertEqual(len(records),20)
            self.assertEqual(records[0]['settings']['Temperature'],3000)
            paths = [Path(tmp)/f'TEST{j:03d}.CR2' for j in range(20)]
            result, labels, stats = pair_raws(records, paths)
            self.assertEqual(stats['matched'],20)
            self.assertEqual(labels[-1]['Temperature'],4900)
            dup, _, counts = pair_raws(records, paths+[Path(tmp)/'other'/'TEST001.CR2'])
            self.assertEqual(len(dup),19)
            self.assertEqual(counts['ambiguous_names'],1)


class FeaturesTests(unittest.TestCase):
    def test_bayer_sensor_features_vary_with_pixels(self):
        import numpy as np
        from rawstyle.features import extract_raw_features
        class DummyRaw:
            raw_pattern = np.array([[0,1],[3,2]], dtype='uint8')
            color_desc = b'RGBG'
            black_level_per_channel = [512,512,512,512]
            white_level = 16383
            camera_whitebalance = np.array([2.2,1.0,1.7,1.0])
            metadata = types.SimpleNamespace(iso_speed=400,shutter=.008,aperture=2.0,focal_len=50)
            def __init__(self, bright):
                self.raw_image_visible=np.full((256,256),int(8192*bright),dtype='uint16')
                self.raw_image_visible[::2,::2]=int(7000*bright)
                self.raw_image_visible[1::2,1::2]=int(4400*bright)
            def __enter__(self): return self
            def __exit__(self,*_): pass
        old=sys.modules.get('rawpy')
        try:
            sys.modules['rawpy']=types.SimpleNamespace(imread=lambda src: DummyRaw(float(src.read().decode())))
            first=extract_raw_features(memoryview(b'0.6'))
            second=extract_raw_features(memoryview(b'1.0'))
            self.assertEqual(first.shape,second.shape)
            self.assertGreater(len(first),180)
            self.assertTrue(np.isfinite(first).all())
            self.assertFalse(np.array_equal(first,second))
        finally:
            if old is None: sys.modules.pop('rawpy',None)
            else: sys.modules['rawpy']=old


class PipelineTests(unittest.TestCase):
    def test_single_hdd_read_into_shared_memory(self):
        from rawstyle.pipeline import _load_into_ram, enumerate_raw_files
        from multiprocessing import shared_memory
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'photo.CR3'
            data=bytes(range(256))*1000
            p.write_bytes(data)
            (Path(tmp)/'not-an-image.txt').write_text('ignore')
            self.assertEqual(enumerate_raw_files(tmp),[p])
            shm,n=_load_into_ram(p)
            try:
                self.assertEqual(bytes(shm.buf[:n]),data)
            finally:
                shm.close();shm.unlink()


class ModelTests(unittest.TestCase):
    def test_kelvin_is_independently_predicted_target(self):
        import numpy as np
        from rawstyle.model import predict, train
        rng=np.random.default_rng(33)
        x=rng.normal(0,1,(52,21)).astype('float32')
        labels=[{'Temperature':float(5200+800*vec[0]),'Tint':float(10+5*vec[1]),
                 'Exposure2012':float(.4+vec[2]/10)} for vec in x]
        model=train(x,labels)
        self.assertIn('Temperature',model['targets'])
        self.assertIn('Tint',model['targets'])
        result=predict(model,x[:3])
        self.assertEqual(result[0]['WhiteBalance'],'Custom')
        self.assertTrue(2000<=result[0]['Temperature']<=50000)
        self.assertNotEqual(result[0]['Temperature'],result[1]['Temperature'])
        self.assertGreaterEqual(model['validation_mae']['Temperature'],0)

    def test_export_keeps_kelvin_and_paths_escaped(self):
        from rawstyle.export import create_predictions_file
        with tempfile.TemporaryDirectory() as tmp:
            fn=create_predictions_file([(Path(tmp)/'My Raw\tFile.CR2',{
                'Temperature':6200,'Tint':-8,'Exposure2012':1.25})],tmp)
            text=fn.read_text()
            self.assertIn('Temperature\tTint',text)
            self.assertIn('6200\t-8\t1.25',text)
            self.assertIn('%09',text)
            self.assertNotIn('WhiteBalance',text)


class BrowserTests(unittest.TestCase):
    def test_local_ui_session_and_status(self):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        from rawstyle.server import App,Handler
        app=App()
        class CustomHandler(Handler): pass
        CustomHandler.app=app
        server=ThreadingHTTPServer(('127.0.0.1',0),CustomHandler)
        runner=threading.Thread(target=server.serve_forever,daemon=True)
        runner.start()
        try:
            url='http://127.0.0.1:'+str(server.server_address[1])
            with urllib.request.urlopen(url+'/') as res:
                html=res.read().decode()
            self.assertIn('<h1>RAWStyle Studio</h1>',html)
            self.assertIn(app.session_token,html)
            with urllib.request.urlopen(url+'/api/status') as res:
                s=json.loads(res.read())
            self.assertIn('model_ready',s)
        finally:
            server.shutdown()
            server.server_close()


if __name__=='__main__': unittest.main()

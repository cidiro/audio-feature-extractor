from mfcc import MFCCExtractor

if __name__ == '__main__':
    mfcc_extractor = MFCCExtractor(
        folder_path=r"C:\Users\Ro2\Desktop\nsynth-test.jsonwav\nsynth-test\audio"
    )
    mfcc_extractor.mean_to_csv("nsynth-test-mean-mfcc.csv")
    mfcc_extractor.std_to_csv("nsynth-test-std-mfcc.csv")

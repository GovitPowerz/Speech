//! Segmenter decision tests (hand-computed from synthetic results rows).

use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{SegmenterConfig, update_segmentation};

fn cfg_no_smooth() -> SegmenterConfig {
    // area 0 so a single crossing triggers; padding/min all 0 so smoothing is a no-op.
    SegmenterConfig {
        rising: 0.5,
        area_rising: 0.0,
        falling: 0.5,
        area_falling: 0.0,
        padding: [0.0; 4],
        min_speech: [0.0; 3],
        min_silence: [0.0; 2],
    }
}

fn spans(s: &Segmentation) -> Vec<(f64, SegClass)> {
    s.segments().iter().map(|x| (x.begin, x.ty)).collect()
}

#[test]
fn clean_rising_falling_crossing() {
    // dt=1.0, offset=0. results cross up between idx1(0.0)->idx2(1.0) at 0.5*dt,
    // and back down between idx3(1.0)->idx4(0.0) at 0.5*dt past idx3. area_rising/falling=0.
    let mut seg = Segmentation::new(5.0);
    let results = vec![0.0, 0.0, 1.0, 1.0, 0.0];
    update_segmentation(
        &mut seg,
        &results,
        SegClass::Speech,
        0.0,
        1.0,
        &cfg_no_smooth(),
    );
    // rising: r(2)>=0.5 && r(1)<0.5 -> begin = 1*(2 - (1.0-0.5)/(1.0-0.0)) = 1.5
    // falling: r(4)<=0.5 && r(3)>0.5 -> end = 1*(4 - (0.0-0.5)/(0.0-1.0)) = 3.5
    let sp = spans(&seg);
    assert_eq!(sp[0], (0.0, SegClass::Other));
    assert_eq!(sp[1], (1.5, SegClass::Speech));
    assert_eq!(sp[2], (3.5, SegClass::Other));
    assert_eq!(*sp.last().unwrap(), (5.0, SegClass::End));
}

#[test]
fn config_parses_seg_golden() {
    let text = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference_data/phase0bii/seg.config"),
    )
    .unwrap();
    let m = speech::legacy_config::parse_legacy_config(&text);
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    // 24-Feb-2014_BLSTM_Spect.config, last-duplicate-key wins: rising 0.6/area 0.05, falling 0.3/area 0.03.
    assert_eq!(cfg.rising, 0.6);
    assert_eq!(cfg.area_rising, 0.05);
    assert_eq!(cfg.falling, 0.3);
    assert_eq!(cfg.area_falling, 0.03);
    assert_eq!(cfg.padding, [0.2, 0.0, 0.3, 0.4]);
    assert_eq!(cfg.min_silence, [0.3, 0.5]);
    assert_eq!(cfg.min_speech, [0.0, 0.3, 0.4]);
}

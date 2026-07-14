//! Documents the WCS sign conventions that platers relies on, verified
//! against the `fitsy` WCS implementation.

use fitsy::{wcs::Wcs, Header};

#[test]
fn test_wcs_library_convention() {
    // Build a simple WCS with known parameters:
    // - Center at (180deg, 45deg)
    // - Reference pixel at FITS (1024, 745) [1-based]
    // - Scale 0.001deg/pixel, rotation 0 (North up, East left)
    let mut header = Header::empty();
    let _ = header.push("NAXIS", 2_i64, None).unwrap();
    let _ = header.push("NAXIS1", 2048_i64, None).unwrap();
    let _ = header.push("NAXIS2", 1490_i64, None).unwrap();
    let _ = header.push("CTYPE1", "RA---TAN", None).unwrap();
    let _ = header.push("CTYPE2", "DEC--TAN", None).unwrap();
    let _ = header.push("CUNIT1", "deg", None).unwrap();
    let _ = header.push("CUNIT2", "deg", None).unwrap();
    let _ = header.push("CRVAL1", 180.0_f64, None).unwrap();
    let _ = header.push("CRVAL2", 45.0_f64, None).unwrap();
    let _ = header.push("CRPIX1", 1024.0_f64, None).unwrap();
    let _ = header.push("CRPIX2", 745.0_f64, None).unwrap();
    let _ = header.push("CD1_1", -0.001_f64, None).unwrap();
    let _ = header.push("CD1_2", 0.0_f64, None).unwrap();
    let _ = header.push("CD2_1", 0.0_f64, None).unwrap();
    let _ = header.push("CD2_2", 0.001_f64, None).unwrap();

    let wcs = Wcs::from_header(&header, ' ')
        .unwrap()
        .expect("header should contain a celestial WCS");

    println!("\n=== Testing WCS conventions (fitsy) ===\n");

    // fitsy pixel coordinates are 0-based; FITS CRPIX is 1-based, so the
    // reference point sits at 0-based (1023, 744).
    let (ref_x, ref_y) = (1023.0_f64, 744.0_f64);

    // Test 1: reference pixel should project to the center.
    let (ra, dec) = wcs.pixel_to_celestial(ref_x, ref_y).unwrap();
    println!("Reference pixel projects to: RA={ra:.6}deg, Dec={dec:.6}deg");
    assert!((ra - 180.0).abs() < 1e-6, "RA at ref pixel: {ra}");
    assert!((dec - 45.0).abs() < 1e-6, "Dec at ref pixel: {dec}");

    // Test 2: moving +100 px in x should decrease RA (East is to the left).
    let (ra_r, dec_r) = wcs.pixel_to_celestial(ref_x + 100.0, ref_y).unwrap();
    println!("Pixel +100 in x projects to: RA={ra_r:.6}deg, Dec={dec_r:.6}deg");
    // CD1_1 = -0.001 deg/px on the tangent plane; on the sphere the RA
    // change at dec=45deg is -0.1 / cos(45deg).
    let expected_ra = 180.0 - 0.1 / 45.0_f64.to_radians().cos();
    assert!((ra_r - expected_ra).abs() < 1e-5, "RA at +100x: {ra_r}");
    assert!((dec_r - 45.0).abs() < 1e-3, "Dec at +100x: {dec_r}");

    // Test 3: moving +100 px in y should increase Dec (so +y points NORTH).
    let (ra_d, dec_d) = wcs.pixel_to_celestial(ref_x, ref_y + 100.0).unwrap();
    println!("Pixel +100 in y projects to: RA={ra_d:.6}deg, Dec={dec_d:.6}deg");
    assert!((ra_d - 180.0).abs() < 1e-6, "RA at +100y: {ra_d}");
    assert!((dec_d - 45.1).abs() < 1e-6, "Dec at +100y: {dec_d}");

    println!("\nConventions confirmed:");
    println!("  - CD1_1 < 0: +x (right) -> -RA (East to the left)");
    println!("  - CD2_2 > 0: +y -> +Dec (so +y points North)");
}

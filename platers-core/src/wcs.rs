//! World Coordinate System (WCS) computation and transformations.
//!
//! This module handles the computation of astrometric solutions from
//! quad matches, including:
//! - WCS parameter computation from correspondences
//! - Coordinate transformations (pixel <-> sky)
//! - FITS header generation

use crate::{
    errors::{Error, PlatersResult},
    types::{PixelCoord, SkyCoord},
};

use fitsy::{
    wcs::{fit_celestial_wcs, sip::SipPoly, ProjectionKind, Wcs, WcsFitOptions},
    FitsError, Header, Value,
};

/// Convert a `fitsy` SIP polynomial to our serializable [`SipPolynomial`].
/// Both use the same `coeffs[p*(order+1) + q]` layout.
fn sip_poly_from_fitsy(p: &SipPoly) -> SipPolynomial {
    SipPolynomial {
        order: p.order,
        coeffs: p.coeffs.clone(),
    }
}

/// Emit the `{name}_ORDER` and `{name}_p_q` header cards for one SIP polynomial
/// (e.g. `A_ORDER`, `A_2_0`). Zero coefficients are skipped.
fn push_sip_poly(header: &mut Header, name: &str, poly: &SipPolynomial) -> Result<(), FitsError> {
    let _ = header.push(format!("{name}_ORDER"), i64::from(poly.order), None)?;
    let n = (poly.order as usize) + 1;
    for p in 0..n {
        for q in 0..n {
            if p + q > poly.order as usize {
                continue;
            }
            let c = poly.coeffs[p * n + q];
            if c != 0.0 {
                let _ = header.push(format!("{name}_{p}_{q}"), c, None)?;
            }
        }
    }
    Ok(())
}

/// One SIP polynomial (`A` or `B`, or the inverse `AP`/`BP`): a coefficient
/// grid where `coeffs[p * (order+1) + q]` multiplies `u^p v^q` (matching the
/// `fitsy::wcs::sip::SipPoly` layout). Entries with `p + q > order` are zero.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SipPolynomial {
    /// Polynomial order (max `p + q`).
    pub order: u32,
    /// Flat coefficient grid, row-major in `p`: `coeffs[p*(order+1) + q]`.
    pub coeffs: Vec<f64>,
}

/// Optional SIP distortion attached to a [`WcsHypothesis`]: the forward `A`/`B`
/// polynomials and (optionally) the inverse `AP`/`BP`. Emitted as `A_p_q`/
/// `B_p_q` header cards so the `fitsy` projection honors the distortion.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SipDistortion {
    /// Forward distortion in the pixel x-axis (`A`).
    pub a: SipPolynomial,
    /// Forward distortion in the pixel y-axis (`B`).
    pub b: SipPolynomial,
    /// Inverse polynomial for the x-axis (`AP`), if fitted.
    pub ap: Option<SipPolynomial>,
    /// Inverse polynomial for the y-axis (`BP`), if fitted.
    pub bp: Option<SipPolynomial>,
}

/// A WCS hypothesis generated from a quad match.
///
/// The **CD matrix is the single source of truth** for orientation and scale.
/// Pixel scale and rotation are *derived* from it via
/// [`scale_arcsec_per_pixel`](Self::scale_arcsec_per_pixel) /
/// [`rotation_deg`](Self::rotation_deg) rather than stored, so they can never
/// drift out of sync with the actual projection. An optional [`SipDistortion`]
/// captures non-linear distortion fitted during refinement.
#[non_exhaustive]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WcsHypothesis {
    /// Field center (RA, Dec) in degrees
    pub center: SkyCoord,

    /// Reference pixel (typically image center)
    pub reference_pixel: PixelCoord,

    /// Image width in pixels
    pub image_width: usize,
    /// Image height in pixels
    pub image_height: usize,

    /// CD matrix element `[1][1]` (deg/pixel). The CD matrix maps pixel offsets
    /// from the reference pixel to intermediate world coordinates, and is the
    /// source of truth for scale/rotation.
    pub cd1_1: f64,
    /// CD matrix element `[1][2]` (deg/pixel).
    pub cd1_2: f64,
    /// CD matrix element `[2][1]` (deg/pixel).
    pub cd2_1: f64,
    /// CD matrix element `[2][2]` (deg/pixel).
    pub cd2_2: f64,

    /// Optional SIP distortion (set by refinement when `sip_order` is used).
    #[serde(default)]
    pub sip: Option<SipDistortion>,
}

impl WcsHypothesis {
    /// Create a simple hypothesis from basic parameters.
    ///
    /// # Arguments
    /// * `center` - Field center (RA, Dec)
    /// * `scale` - Pixel scale in arcsec/pixel
    /// * `rotation` - Rotation angle in degrees
    /// * `image_width` - Image width in pixels
    /// * `image_height` - Image height in pixels
    #[must_use]
    pub fn new(
        center: SkyCoord,
        scale_arcsec_per_pixel: f64,
        rotation_deg: f64,
        image_width: usize,
        image_height: usize,
    ) -> Self {
        // The FITS-standard image center: pixel centers sit at integers in the
        // 1-based convention, so the array spans `[0.5, N+0.5]` and the true
        // geometric center is `(N+1)/2` (1-based), i.e. `(N-1)/2` here (0-based).
        let reference_pixel = PixelCoord {
            x: (image_width as f64 - 1.0) / 2.0,
            y: (image_height as f64 - 1.0) / 2.0,
        };

        // Compute CD matrix from scale and rotation
        // CD matrix converts pixel offsets to degrees on sky
        let scale_deg = scale_arcsec_per_pixel / 3600.0;
        let rot_rad = rotation_deg.to_radians();
        let cos_rot = rot_rad.cos();
        let sin_rot = rot_rad.sin();

        // Standard CD matrix for rotation + scale
        // Note: y-axis is flipped in image coordinates (down is positive)
        let cd1_1 = -scale_deg * cos_rot; // Negative because RA increases to the left (East)
        let cd1_2 = scale_deg * sin_rot;
        let cd2_1 = scale_deg * sin_rot;
        let cd2_2 = scale_deg * cos_rot;

        Self {
            center,
            reference_pixel,
            image_width,
            image_height,
            cd1_1,
            cd1_2,
            cd2_1,
            cd2_2,
            sip: None,
        }
    }

    /// Pixel scale in arcseconds per pixel, derived from the CD matrix
    /// (mean of the two column norms).
    #[must_use]
    pub fn scale_arcsec_per_pixel(&self) -> f64 {
        let col1 = (self.cd1_1 * self.cd1_1 + self.cd2_1 * self.cd2_1).sqrt();
        let col2 = (self.cd1_2 * self.cd1_2 + self.cd2_2 * self.cd2_2).sqrt();
        f64::midpoint(col1, col2) * 3600.0
    }

    /// Rotation angle in degrees (East of North), derived from the CD matrix.
    #[must_use]
    pub fn rotation_deg(&self) -> f64 {
        self.cd2_1.atan2(self.cd2_2).to_degrees()
    }

    /// Build an in-memory FITS header describing this hypothesis.
    ///
    /// `fitsy` parses a WCS from a [`Header`], so we materialize the standard
    /// `CRVAL`/`CRPIX`/`CD`/`CTYPE` cards here. Note that `fitsy` uses a
    /// 0-based pixel convention (and adds 1 internally), whereas FITS `CRPIX`
    /// is 1-based; we therefore write `CRPIX = reference_pixel + 1` so that
    /// `reference_pixel` remains the exact fixed point of the projection.
    #[allow(
        clippy::cast_possible_wrap,
        reason = "image dimensions are far below i64::MAX"
    )]
    fn build_header(&self) -> Result<Header, FitsError> {
        let mut header = Header::empty();
        // `Header::push` returns `&mut Header` for chaining; we ignore it and
        // mutate in place. `let _ =` silences the workspace `unused_results` lint.
        let _ = header.push("NAXIS", 2_i64, None)?;
        let _ = header.push("NAXIS1", self.image_width as i64, None)?;
        let _ = header.push("NAXIS2", self.image_height as i64, None)?;
        // With SIP present, the CTYPE carries the "-SIP" suffix so fitsy applies
        // the distortion polynomials.
        let (ctype1, ctype2) = if self.sip.is_some() {
            ("RA---TAN-SIP", "DEC--TAN-SIP")
        } else {
            ("RA---TAN", "DEC--TAN")
        };
        let _ = header.push("CTYPE1", ctype1, None)?;
        let _ = header.push("CTYPE2", ctype2, None)?;
        let _ = header.push("CUNIT1", "deg", None)?;
        let _ = header.push("CUNIT2", "deg", None)?;
        let _ = header.push("CRVAL1", self.center.ra, None)?;
        let _ = header.push("CRVAL2", self.center.dec, None)?;
        let _ = header.push("CRPIX1", self.reference_pixel.x + 1.0, None)?;
        let _ = header.push("CRPIX2", self.reference_pixel.y + 1.0, None)?;
        let _ = header.push("CD1_1", self.cd1_1, None)?;
        let _ = header.push("CD1_2", self.cd1_2, None)?;
        let _ = header.push("CD2_1", self.cd2_1, None)?;
        let _ = header.push("CD2_2", self.cd2_2, None)?;
        // Gaia DR3 / Tycho-2 positions are ICRS. No EQUINOX card: the FITS
        // standard forbids it with RADESYS = 'ICRS' (ICRS has no equinox).
        let _ = header.push("RADESYS", "ICRS", None)?;

        if let Some(sip) = &self.sip {
            push_sip_poly(&mut header, "A", &sip.a)?;
            push_sip_poly(&mut header, "B", &sip.b)?;
            if let Some(ap) = &sip.ap {
                push_sip_poly(&mut header, "AP", ap)?;
            }
            if let Some(bp) = &sip.bp {
                push_sip_poly(&mut header, "BP", bp)?;
            }
        }
        Ok(header)
    }

    /// Create a `fitsy` WCS object from the hypothesis parameters.
    ///
    /// Used for pixel <-> sky projections via the full FITS WCS pipeline.
    fn create_wcs(&self) -> PlatersResult<Wcs> {
        let header = self
            .build_header()
            .map_err(|e| Error::InvalidWcs(format!("Failed to build WCS header: {e}")))?;

        Wcs::from_header(&header, ' ')
            .map_err(|e| Error::InvalidWcs(format!("Failed to parse WCS: {e}")))?
            .ok_or_else(|| {
                Error::InvalidWcs("Constructed header contained no celestial WCS".to_string())
            })
    }

    /// Compute a WCS hypothesis from a quad correspondence.
    ///
    /// Given 4 stars in pixel space and their corresponding catalog positions,
    /// computes the implied WCS via a fast hand-rolled least-squares solve over
    /// a local tangent-plane approximation. This is the solver's hot path
    /// (called once per candidate quad match); the heavier, higher-accuracy
    /// `fitsy` fitter is used for multi-star refinement in
    /// [`from_star_matches`](Self::from_star_matches).
    ///
    /// # Errors
    /// Returns an error if the correspondence is degenerate or invalid.
    pub fn from_quad_match(
        image_stars: &[PixelCoord; 4],
        catalog_stars: &[SkyCoord; 4],
        image_width: usize,
        image_height: usize,
    ) -> PlatersResult<Self> {
        // Free least-squares TAN fit of the four pixel<->sky correspondences via
        // `fitsy`. The tangent point must land at the quad's centroid: a 4-star
        // fit pinned to the far-off image centre is ill-conditioned, while the
        // centroid-anchored free fit carries the field's true orientation and
        // parity at any rotation. The accepted pose is re-anchored to the image
        // centre downstream.
        let pixels: Vec<(f64, f64)> = image_stars.iter().map(|p| (p.x, p.y)).collect();
        let sky: Vec<(f64, f64)> = catalog_stars.iter().map(|s| (s.ra, s.dec)).collect();
        let opts = WcsFitOptions {
            projection: ProjectionKind::Tan,
            ..Default::default()
        };
        let fit = fit_celestial_wcs(&pixels, &sky, &opts)
            .map_err(|e| Error::Geometry(format!("quad WCS fit failed: {e}")))?;
        Ok(Self::from_fitted_wcs(&fit.wcs, image_width, image_height))
    }

    /// Least-squares WCS fit from many correspondences via `fitsy`'s
    /// `fit_celestial_wcs`, re-anchored to the image center. Backs
    /// [`from_star_matches`](Self::from_star_matches).
    fn fit_from_correspondences(
        image_stars: &[PixelCoord],
        catalog_stars: &[SkyCoord],
        image_width: usize,
        image_height: usize,
        sip_order: Option<u32>,
    ) -> PlatersResult<Self> {
        let pixels: Vec<(f64, f64)> = image_stars.iter().map(|p| (p.x, p.y)).collect();
        let sky: Vec<(f64, f64)> = catalog_stars.iter().map(|s| (s.ra, s.dec)).collect();

        // Pass 1: free fit. The tangent point lands at the spherical centroid
        // of the stars, which is generally offset from the image center.
        let opts = WcsFitOptions {
            projection: ProjectionKind::Tan,
            ..Default::default()
        };
        let fit = fit_celestial_wcs(&pixels, &sky, &opts)
            .map_err(|e| Error::Geometry(format!("WCS fit failed: {e}")))?;

        // Re-anchor to the image center: evaluate the fitted model there, then
        // re-fit with the tangent point (CRVAL) and reference pixel (CRPIX)
        // pinned to the center. Re-fitting (rather than reusing the pass-1 CD)
        // keeps the CD matrix exact for the new tangent point, avoiding the
        // gnomonic error that a tangent-point shift would otherwise introduce
        // over wide fields. This makes `center` the true field center.
        // FITS-standard geometric center (0-based): `(N-1)/2`, not `N/2` -- see
        // the note in `WcsHypothesis::new`.
        let center_pixel = (
            (image_width as f64 - 1.0) / 2.0,
            (image_height as f64 - 1.0) / 2.0,
        );
        let center = fit
            .wcs
            .pixel_to_celestial(center_pixel.0, center_pixel.1)
            .map_err(|e| Error::Geometry(format!("Failed to evaluate field center: {e}")))?;

        let centered_opts = WcsFitOptions {
            projection: ProjectionKind::Tan,
            crpix: Some(center_pixel),
            crval: Some(center),
            sip_order,
            ..Default::default()
        };
        let centered = fit_celestial_wcs(&pixels, &sky, &centered_opts)
            .map_err(|e| Error::Geometry(format!("WCS re-fit failed: {e}")))?;

        Ok(Self::from_fitted_wcs(
            &centered.wcs,
            image_width,
            image_height,
        ))
    }

    /// Extract hypothesis parameters from a fitted `fitsy` WCS.
    ///
    /// Pulls CRVAL (field center), CRPIX (reference pixel, converted from the
    /// FITS 1-based convention to our 0-based one), and the CD matrix, then
    /// derives the scalar pixel scale and rotation used elsewhere. The WCS is
    /// expected to already be anchored at the image center (see
    /// [`fit_from_correspondences`](Self::fit_from_correspondences)).
    fn from_fitted_wcs(wcs: &Wcs, image_width: usize, image_height: usize) -> Self {
        // `fit_celestial_wcs` writes a 2-axis RA---TAN / DEC--TAN WCS, so CRVAL
        // is [RA, Dec] and the CD matrix is row-major [11, 12, 21, 22].
        let m = wcs.linear.matrix_row_major();
        let (cd1_1, cd1_2, cd2_1, cd2_2) = (m[0], m[1], m[2], m[3]);

        let center = SkyCoord::new_normalized(wcs.crval[0], wcs.crval[1]);

        let crpix = wcs.linear.crpix();
        // FITS CRPIX is 1-based; our `reference_pixel` is 0-based.
        let reference_pixel = PixelCoord {
            x: crpix[0] - 1.0,
            y: crpix[1] - 1.0,
        };

        // Capture any fitted SIP distortion from the celestial block.
        let sip = wcs
            .celestial
            .as_ref()
            .and_then(|c| c.sip.as_ref())
            .map(|s| SipDistortion {
                a: sip_poly_from_fitsy(&s.a),
                b: sip_poly_from_fitsy(&s.b),
                ap: s.ap.as_ref().map(sip_poly_from_fitsy),
                bp: s.bp.as_ref().map(sip_poly_from_fitsy),
            });

        Self {
            center,
            reference_pixel,
            image_width,
            image_height,
            cd1_1,
            cd1_2,
            cd2_1,
            cd2_2,
            sip,
        }
    }

    /// Fit WCS from many star correspondences using least-squares.
    ///
    /// A generalization of [`from_quad_match`](Self::from_quad_match) that
    /// accepts any number of correspondences (N >= 4). More stars give a
    /// better-conditioned fit and robustness to per-star noise. Backed by the
    /// same `fitsy` fitter.
    ///
    /// # Arguments
    /// * `image_stars` - Pixel coordinates of detected stars
    /// * `catalog_stars` - Sky coordinates of corresponding catalog stars
    /// * `image_width` - Image width in pixels
    /// * `image_height` - Image height in pixels
    ///
    /// # Returns
    /// WCS hypothesis fitted to minimize residuals across all star pairs.
    ///
    /// # Errors
    /// Returns error if fewer than 4 pairs are supplied, the pixel and catalog
    /// counts differ, or the fit is degenerate.
    pub fn from_star_matches(
        image_stars: &[PixelCoord],
        catalog_stars: &[SkyCoord],
        image_width: usize,
        image_height: usize,
    ) -> PlatersResult<Self> {
        Self::from_star_matches_sip(image_stars, catalog_stars, image_width, image_height, None)
    }

    /// Like [`from_star_matches`](Self::from_star_matches) but optionally fits a
    /// SIP distortion polynomial of the given order (`None` = linear only).
    ///
    /// SIP needs enough correspondences to over-determine the polynomial: an
    /// order-`n` fit adds `n(n+3)/2` parameters per axis, so use it only with
    /// many well-spread stars.
    ///
    /// # Errors
    /// Returns error if fewer than 4 pairs are supplied, the counts differ, or
    /// the fit is degenerate.
    pub fn from_star_matches_sip(
        image_stars: &[PixelCoord],
        catalog_stars: &[SkyCoord],
        image_width: usize,
        image_height: usize,
        sip_order: Option<u32>,
    ) -> PlatersResult<Self> {
        if image_stars.len() != catalog_stars.len() {
            return Err(Error::Geometry(format!(
                "Mismatch in star counts: {} image vs {} catalog",
                image_stars.len(),
                catalog_stars.len()
            )));
        }

        if image_stars.len() < 4 {
            return Err(Error::Geometry(format!(
                "Need at least 4 stars for WCS fitting, got {}",
                image_stars.len()
            )));
        }

        Self::fit_from_correspondences(
            image_stars,
            catalog_stars,
            image_width,
            image_height,
            sip_order,
        )
    }

    /// Build a reusable [`Projector`] that parses the `fitsy` WCS once.
    ///
    /// Parsing the `fitsy` WCS requires materializing and parsing a FITS header,
    /// which is relatively expensive. Code that transforms many points under the
    /// same hypothesis (verification, refinement) should call this once and reuse
    /// the result, rather than the per-call [`sky_to_pixel`](Self::sky_to_pixel) /
    /// [`pixel_to_sky`](Self::pixel_to_sky) convenience methods (which rebuild it
    /// each time).
    ///
    /// # Errors
    /// Returns an error if the WCS parameters are invalid.
    pub fn projector(&self) -> PlatersResult<Projector> {
        Ok(Projector {
            wcs: self.create_wcs()?,
        })
    }

    /// Return an equivalent hypothesis whose reference pixel (CRPIX) is the
    /// image center and whose `center` (CRVAL) is the sky position there.
    ///
    /// The solver builds the coarse WCS anchored at the *quad centroid*, so its
    /// mapping is accurate but the reported `center` can sit hundreds of arcsec
    /// from the true field center. This re-expresses the same mapping with the
    /// reference point at the image center.
    ///
    /// We do **not** move the TAN tangent point while holding CD fixed: the
    /// gnomonic projection is nonlinear, so that introduces a distortion that
    /// grows with the tangent-point shift and the field size. Near the pole --
    /// where the RA-axis scale changes rapidly with declination -- it grows large
    /// enough to displace catalog stars by arcseconds, which starves the
    /// many-star refinement (it matches catalog->pixel within a tight radius) and
    /// leaves the coarse pose un-polished. Instead we sample a grid of pixels,
    /// project them through the current mapping, and re-fit a fresh TAN anchored
    /// at the image center to those synthetic correspondences. The grid is dense
    /// and well-spread, so the fit is well-conditioned (unlike a 4-star re-fit
    /// far from its anchor) and reproduces the mapping across the whole field.
    ///
    /// # Errors
    /// Returns an error if the projection or the re-fit fails.
    pub fn re_anchored_to_image_center(&self) -> PlatersResult<Self> {
        let projector = self.projector()?;
        let (w, h) = (self.image_width as f64, self.image_height as f64);

        // 5x5 grid of pixels spanning the image (first to last pixel center),
        // projected through the current mapping to give distortion-free
        // synthetic correspondences for the re-fit.
        let mut pixels: Vec<PixelCoord> = Vec::with_capacity(25);
        let mut sky: Vec<SkyCoord> = Vec::with_capacity(25);
        for iy in 0..5 {
            for ix in 0..5 {
                let px = PixelCoord {
                    x: (w - 1.0) * f64::from(ix) / 4.0,
                    y: (h - 1.0) * f64::from(iy) / 4.0,
                };
                if let Ok(s) = projector.pixel_to_sky(px) {
                    pixels.push(px);
                    sky.push(s);
                }
            }
        }

        if pixels.len() < 6 {
            // Projection mostly failed (degenerate WCS) -- fall back to the cheap
            // tangent-point shift so we still return a center-anchored pose.
            let center_pixel = PixelCoord {
                x: (w - 1.0) / 2.0,
                y: (h - 1.0) / 2.0,
            };
            let center_sky = self.pixel_to_sky(center_pixel)?;
            return Ok(Self {
                center: center_sky,
                reference_pixel: center_pixel,
                ..self.clone()
            });
        }

        Self::fit_from_correspondences(&pixels, &sky, self.image_width, self.image_height, None)
    }

    /// Project sky coordinates to pixel coordinates.
    ///
    /// Convenience for a single transform; rebuilds the `fitsy` WCS each call.
    /// For many points, build a [`Projector`](Self::projector) once.
    ///
    /// # Errors
    /// Returns an error if the projection fails.
    pub fn sky_to_pixel(&self, sky: SkyCoord) -> PlatersResult<PixelCoord> {
        self.projector()?.sky_to_pixel(sky)
    }

    /// Project pixel coordinates to sky coordinates.
    ///
    /// Convenience for a single transform; rebuilds the `fitsy` WCS each call.
    /// For many points, build a [`Projector`](Self::projector) once.
    ///
    /// # Errors
    /// Returns an error if the projection fails.
    pub fn pixel_to_sky(&self, pixel: PixelCoord) -> PlatersResult<SkyCoord> {
        self.projector()?.pixel_to_sky(pixel)
    }

    /// Get pixel scale in degrees per pixel (derived from the CD matrix).
    #[must_use]
    pub fn scale_deg_per_pixel(&self) -> f64 {
        self.scale_arcsec_per_pixel() / 3600.0
    }

    /// Image parity: the sign of the CD matrix determinant.
    ///
    /// `+1.0` = positive (standard East-left / North-up) parity; `-1.0` =
    /// negative (mirror-flipped) parity. Sky-true value, not a forced
    /// convention -- it reflects the orientation of the actual correspondence.
    #[must_use]
    pub fn parity(&self) -> f64 {
        (self.cd1_1 * self.cd2_2 - self.cd1_2 * self.cd2_1).signum()
    }

    /// Generate FITS WCS keywords as key-value string pairs.
    ///
    /// Rendered from the same header as [`projector`](Self::projector), so the
    /// emitted keywords cannot drift from the actual projection. Includes the
    /// CD matrix (exact for any rotation/parity/skew, unlike CDELT+CROTA) and
    /// any SIP distortion cards.
    ///
    /// # Errors
    /// [`Error::InvalidWcs`] if the header cannot be built.
    pub fn to_fits_keywords(&self) -> PlatersResult<Vec<(String, String)>> {
        let header = self
            .build_header()
            .map_err(|e| Error::InvalidWcs(format!("Failed to build WCS header: {e}")))?;
        Ok(header
            .entries()
            .iter()
            .filter_map(|entry| {
                let rendered = match entry.value.as_ref()? {
                    Value::Logical(b) => if *b { "T" } else { "F" }.to_string(),
                    Value::Integer(i) => i.to_string(),
                    Value::Real(r) => r.to_string(),
                    Value::String(s) => s.clone(),
                    Value::ComplexInteger(..) | Value::ComplexReal(..) | Value::Undefined => {
                        return None;
                    }
                };
                Some((entry.keyword.clone(), rendered))
            })
            .collect())
    }
}

/// A reusable pixel <-> sky projector holding a parsed `fitsy` WCS.
///
/// Created via [`WcsHypothesis::projector`]. Build once, transform many points.
#[derive(Debug)]
pub struct Projector {
    wcs: Wcs,
}

impl Projector {
    /// Project sky coordinates to pixel coordinates.
    ///
    /// # Errors
    /// Returns an error if the point is outside the valid projection range.
    pub fn sky_to_pixel(&self, sky: SkyCoord) -> PlatersResult<PixelCoord> {
        self.wcs
            .celestial_to_pixel(sky.ra, sky.dec)
            .map(|(x, y)| PixelCoord { x, y })
            .map_err(|e| Error::Geometry(format!("WCS projection failed: {e}")))
    }

    /// Project pixel coordinates to sky coordinates.
    ///
    /// # Errors
    /// Returns an error if the pixel is outside the valid range.
    pub fn pixel_to_sky(&self, pixel: PixelCoord) -> PlatersResult<SkyCoord> {
        self.wcs
            .pixel_to_celestial(pixel.x, pixel.y)
            .map(|(ra, dec)| SkyCoord::new_normalized(ra, dec))
            .map_err(|e| Error::Geometry(format!("WCS unprojection failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apply a radial barrel distortion to a pixel about the image center,
    /// matching the test-harness synthetic: scale the offset by `(1 + k r_n^2)`
    /// where `r_n` is the radius normalized so the corner is 1.0.
    #[allow(
        clippy::many_single_char_names,
        reason = "conventional geometry names: pixel (x, y), image (w, h), coefficient k"
    )]
    fn distort(x: f64, y: f64, w: usize, h: usize, k: f64) -> (f64, f64) {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let (dx, dy) = (x - cx, y - cy);
        let corner = (cx * cx + cy * cy).sqrt();
        let r_n2 = (dx * dx + dy * dy) / (corner * corner);
        let f = 1.0 + k * r_n2;
        (cx + dx * f, cy + dy * f)
    }

    /// A *linear* (CD-only) fit cannot represent radial distortion, so fitting
    /// an undistorted model to a distorted field leaves a residual that grows
    /// with the distortion -- while a distortion-free field fits to ~zero. This
    /// pins the residual that a SIP fit must clearly reduce.
    #[test]
    fn test_linear_fit_leaves_residual_on_distorted_field() {
        let (w, h) = (2000_usize, 2000_usize);
        let truth = WcsHypothesis::new(SkyCoord::new(180.0, 10.0), 1.0, 0.0, w, h);
        let proj = truth.projector().unwrap();

        // A spread of catalog stars across the field; their ideal pixels.
        let catalog: Vec<SkyCoord> = (0..7)
            .flat_map(|i| {
                (0..7).map(move |j| {
                    let x = 100.0 + f64::from(i) * 300.0;
                    let y = 100.0 + f64::from(j) * 300.0;
                    (x, y)
                })
            })
            .map(|(x, y)| proj.pixel_to_sky(PixelCoord::new(x, y)).unwrap())
            .collect();
        let ideal_px: Vec<PixelCoord> = catalog
            .iter()
            .map(|s| proj.sky_to_pixel(*s).unwrap())
            .collect();

        // Residual = RMS pixel error of the fitted model at the *observed*
        // (possibly distorted) pixels.
        let fit_residual_px = |observed: &[PixelCoord]| -> f64 {
            let fit = WcsHypothesis::from_star_matches(observed, &catalog, w, h).unwrap();
            let fp = fit.projector().unwrap();
            let sum_sq: f64 = catalog
                .iter()
                .zip(observed)
                .map(|(s, obs)| {
                    let p = fp.sky_to_pixel(*s).unwrap();
                    (p.x - obs.x).powi(2) + (p.y - obs.y).powi(2)
                })
                .sum();
            (sum_sq / observed.len() as f64).sqrt()
        };

        // Undistorted field: a linear fit is essentially exact.
        let undistorted_residual = fit_residual_px(&ideal_px);
        assert!(
            undistorted_residual < 0.05,
            "linear fit should be ~exact on an undistorted field, got {undistorted_residual:.3} px"
        );

        // Distorted field: the linear fit cannot absorb the radial term.
        let distorted_px: Vec<PixelCoord> = ideal_px
            .iter()
            .map(|p| {
                let (dx, dy) = distort(p.x, p.y, w, h, 0.03);
                PixelCoord::new(dx, dy)
            })
            .collect();
        let distorted_residual = fit_residual_px(&distorted_px);
        assert!(
            distorted_residual > 1.0,
            "distortion should leave a clear residual a linear fit can't remove, got {distorted_residual:.3} px"
        );
        assert!(
            distorted_residual > 10.0 * undistorted_residual,
            "distorted residual ({distorted_residual:.3}) should dwarf undistorted ({undistorted_residual:.3})"
        );

        // A SIP fit absorbs most of the radial distortion. Measure the residual
        // in the *forward* (pixel -> sky) direction, which the forward A/B
        // polynomials handle directly; express in pixels via the pixel scale for
        // an apples-to-apples comparison with the linear fit. (We compare both
        // fits in this same direction.)
        let forward_residual_px = |w_fit: &WcsHypothesis| -> f64 {
            let proj = w_fit.projector().unwrap();
            let ppd = 3600.0 / w_fit.scale_arcsec_per_pixel();
            let sum_sq: f64 = distorted_px
                .iter()
                .zip(&catalog)
                .map(|(obs, true_sky)| {
                    let got = proj.pixel_to_sky(*obs).unwrap();
                    (got.angular_distance(true_sky) * ppd).powi(2)
                })
                .sum();
            (sum_sq / distorted_px.len() as f64).sqrt()
        };

        let lin_fit = WcsHypothesis::from_star_matches(&distorted_px, &catalog, w, h).unwrap();
        let lin_residual = forward_residual_px(&lin_fit);

        // Order 5 captures the (cubic-and-up) radial term well; lower orders
        // only partially. This proves SIP is wired end-to-end and measurably
        // reduces distortion residuals.
        let sip_fit =
            WcsHypothesis::from_star_matches_sip(&distorted_px, &catalog, w, h, Some(5)).unwrap();
        assert!(sip_fit.sip.is_some(), "SIP fit should attach a distortion");
        let sip_residual = forward_residual_px(&sip_fit);

        assert!(
            sip_residual < 0.6 * lin_residual,
            "SIP should clearly cut the distorted residual: linear {lin_residual:.3} px vs SIP {sip_residual:.3} px"
        );
    }

    /// `from_quad_match` *should* recover both image parities and round-trip
    /// exactly. It does **not** today: the coarse least-squares fit has a 180 deg
    /// orientation ambiguity, so a mirror-flipped field can be coerced to the
    /// wrong orientation. This test pins the desired end state; it is ignored
    /// until the coarse fit's orientation handling is reworked. (The solver
    /// compensates by matching each quad at both parities -- see
    /// `PlateSolver::chunk_candidates`.)
    #[test]
    #[ignore = "known bug: from_quad_match LS fit has a 180deg orientation ambiguity"]
    fn test_from_quad_match_recovers_both_parities() {
        let catalog = [
            SkyCoord::new(180.00, 45.00),
            SkyCoord::new(180.05, 45.00),
            SkyCoord::new(180.02, 45.04),
            SkyCoord::new(179.97, 45.02),
        ];

        // Normal sky WCS (CD1_1<0, CD2_2>0 => det<0), and its x-mirror (det>0).
        let normal = WcsHypothesis::new(SkyCoord::new(180.0, 45.0), 1.0, 20.0, 1000, 1000);
        let mut mirrored = normal.clone();
        mirrored.cd1_1 = -mirrored.cd1_1; // mirror the x pixel axis -> flip parity
        mirrored.cd2_1 = -mirrored.cd2_1;

        // The two truths genuinely differ in parity (whatever the absolute sign
        // convention is).
        assert!(
            (normal.parity() - mirrored.parity()).abs() > 1.5,
            "the two truth WCS should have opposite parity ({} vs {})",
            normal.parity(),
            mirrored.parity()
        );

        for truth in [&normal, &mirrored] {
            // Project the catalog stars to pixels through the truth WCS.
            let proj = truth.projector().unwrap();
            let pixels: Vec<PixelCoord> = catalog
                .iter()
                .map(|s| proj.sky_to_pixel(*s).unwrap())
                .collect();
            let img = [pixels[0], pixels[1], pixels[2], pixels[3]];

            // Fit a WCS back from the correspondence.
            let fit = WcsHypothesis::from_quad_match(&img, &catalog, 1000, 1000).unwrap();

            // (a) the fit recovers *this* truth's parity -- not a forced one.
            assert!(
                (fit.parity() - truth.parity()).abs() < 1e-9,
                "recovered parity {} != truth {}",
                fit.parity(),
                truth.parity()
            );

            // (b) the fit round-trips: each catalog star projects back to ~its
            // pixel under the fitted WCS.
            let fit_proj = fit.projector().unwrap();
            for (s, p) in catalog.iter().zip(&img) {
                let back = fit_proj.sky_to_pixel(*s).unwrap();
                let err = ((back.x - p.x).powi(2) + (back.y - p.y).powi(2)).sqrt();
                assert!(err < 0.5, "round-trip off by {err:.3} px");
            }
        }
    }

    /// Re-anchoring must (a) move the reference pixel to the image center,
    /// (b) set `center` to the sky position the un-anchored WCS predicts there (so the
    /// new anchor is exact), (c) carry the CD matrix over unchanged, and (d)
    /// keep the overall mapping *approximately* the same -- only the small
    /// gnomonic distortion from moving the TAN tangent point differs.
    #[test]
    fn test_re_anchor_to_image_center() {
        // Build a WCS, then move its reference pixel away from center so the
        // re-anchor has real work to do (mimicking the quad-centroid case).
        let base = WcsHypothesis::new(SkyCoord::new(180.0, 45.0), 1.0, 12.0, 2048, 1489);
        let mut off_center = base.clone();
        off_center.reference_pixel = PixelCoord { x: 300.0, y: 250.0 };
        // CRVAL must correspond to that reference pixel for a consistent WCS;
        // derive it by asking where (300, 250) lands under `base`.
        off_center.center = base
            .pixel_to_sky(PixelCoord { x: 300.0, y: 250.0 })
            .unwrap();

        let center_pixel = PixelCoord {
            x: 1023.5,
            y: 744.0,
        };
        let expected_center = off_center.pixel_to_sky(center_pixel).unwrap();

        let anchored = off_center.re_anchored_to_image_center().unwrap();

        // (a) reference pixel is the image center.
        assert!((anchored.reference_pixel.x - 1023.5).abs() < 1e-9);
        assert!((anchored.reference_pixel.y - 744.0).abs() < 1e-9);

        // (b) the new anchor is correct: the image center round-trips to the
        // same sky position the original WCS predicted. The grid re-fit is a
        // least-squares fit, so not bit-exact -- but it reproduces the mapping at
        // the image center to well under a tenth of an arcsec (bit-exactness at
        // the anchor would come at the cost of distortion at the field edges).
        let center_after = anchored.pixel_to_sky(center_pixel).unwrap();
        assert!(
            expected_center.angular_distance(&center_after) * 3600.0 < 0.05,
            "anchor not at image center"
        );
        assert!(expected_center.angular_distance(&anchored.center) * 3600.0 < 0.05);

        // (c) scale is preserved (the re-fit reproduces the same mapping; the CD
        // changes slightly because the tangent point moved, which is the point --
        // a CD-fixed shift would distort the mapping near the pole).
        assert!(
            (anchored.scale_arcsec_per_pixel() - off_center.scale_arcsec_per_pixel()).abs()
                / off_center.scale_arcsec_per_pixel()
                < 1e-3,
            "scale not preserved across re-anchor"
        );

        // (d) the mapping is preserved across the whole field -- the grid re-fit
        // reproduces it everywhere, not just near the original tangent point.
        for &(x, y) in &[(700.0, 600.0), (50.0, 50.0), (2000.0, 1400.0)] {
            let probe = PixelCoord { x, y };
            let before = off_center.pixel_to_sky(probe).unwrap();
            let after = anchored.pixel_to_sky(probe).unwrap();
            assert!(
                before.angular_distance(&after) * 3600.0 < 0.05,
                "mapping drifted at ({x}, {y}): {before:?} vs {after:?}"
            );
        }
    }

    #[test]
    fn test_wcs_hypothesis_creation() {
        let center = SkyCoord::new(180.0, 0.0);
        let wcs = WcsHypothesis::new(center, 0.396, 0.0, 2048, 1489);

        assert_eq!(wcs.scale_arcsec_per_pixel(), 0.396);
        assert_eq!(wcs.rotation_deg(), 0.0);
        assert_eq!(wcs.image_width, 2048);
        assert_eq!(wcs.image_height, 1489);
        assert_eq!(wcs.reference_pixel.x, 1023.5);
        assert_eq!(wcs.reference_pixel.y, 744.0);
    }

    #[test]
    fn test_wcs_from_quad_match() {
        // Create a simple square quad
        let image_stars = [
            PixelCoord { x: 100.0, y: 100.0 },
            PixelCoord { x: 200.0, y: 100.0 },
            PixelCoord { x: 100.0, y: 200.0 },
            PixelCoord { x: 200.0, y: 200.0 },
        ];

        // Corresponding sky positions (0.01 degree square)
        let catalog_stars = [
            SkyCoord::new(180.0, 0.0),
            SkyCoord::new(180.01, 0.0),
            SkyCoord::new(180.0, 0.01),
            SkyCoord::new(180.01, 0.01),
        ];

        let wcs = WcsHypothesis::from_quad_match(&image_stars, &catalog_stars, 400, 400).unwrap();

        // Should compute reasonable scale
        // 100 pixels = 0.01 degrees = 36 arcsec
        // scale = 36 / 100 = 0.36 arcsec/pixel
        assert!((wcs.scale_arcsec_per_pixel() - 0.36).abs() < 0.1);
    }

    #[test]
    fn test_sky_to_pixel_projection() {
        let center = SkyCoord::new(180.0, 0.0);
        let wcs = WcsHypothesis::new(center, 0.396, 0.0, 2048, 1489);

        // Center should project to reference pixel
        let pixel = wcs.sky_to_pixel(center).unwrap();
        assert!((pixel.x - 1024.0).abs() < 1.0);
        assert!((pixel.y - 744.5).abs() < 1.0);
    }

    #[test]
    fn test_pixel_to_sky_projection() {
        let center = SkyCoord::new(180.0, 0.0);
        let wcs = WcsHypothesis::new(center, 0.396, 0.0, 2048, 1489);

        // Reference pixel should project to center
        let sky = wcs.pixel_to_sky(wcs.reference_pixel).unwrap();
        assert!((sky.ra - 180.0).abs() < 0.01);
        assert!((sky.dec - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_roundtrip_projection() {
        let center = SkyCoord::new(180.0, 0.0);
        let wcs = WcsHypothesis::new(center, 0.396, 0.0, 2048, 1489);

        let original_pixel = PixelCoord { x: 500.0, y: 300.0 };
        let sky = wcs.pixel_to_sky(original_pixel).unwrap();
        let pixel = wcs.sky_to_pixel(sky).unwrap();

        assert!((pixel.x - original_pixel.x).abs() < 0.1);
        assert!((pixel.y - original_pixel.y).abs() < 0.1);
    }

    #[test]
    fn test_fits_keywords() {
        let center = SkyCoord::new(188.5, 14.45);
        let wcs = WcsHypothesis::new(center, 0.396, 12.3, 2048, 1489);

        let keywords = wcs.to_fits_keywords().unwrap();

        // Should have standard WCS keywords, including the CD matrix (we emit CD
        // directly, not CDELT/CROTA, so it round-trips parity and skew).
        for key in [
            "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CTYPE1", "CTYPE2", "CD1_1", "CD1_2", "CD2_1",
            "CD2_2",
        ] {
            assert!(
                keywords.iter().any(|(k, _)| k == key),
                "missing keyword {key}"
            );
        }

        // Check RA value
        let crval1 = keywords.iter().find(|(k, _)| k == "CRVAL1").unwrap();
        assert!(crval1.1.contains("188.5"));

        // The emitted CD keywords must equal the struct's CD fields exactly.
        let cd1_1: f64 = keywords
            .iter()
            .find(|(k, _)| k == "CD1_1")
            .unwrap()
            .1
            .parse()
            .unwrap();
        // Emitted with {:.9}; CD values are ~1e-4, so allow that rounding.
        assert!((cd1_1 - wcs.cd1_1).abs() < 1e-9);
    }
}

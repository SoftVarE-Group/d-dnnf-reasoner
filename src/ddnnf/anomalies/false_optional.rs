use crate::Ddnnf;

 impl Ddnnf {
    /// Compute all false-optional features
    /// A feature is false optional iff there is no valid configuration that excludes
    /// the feature while simultanously excluding its parent
    #[allow(dead_code)]
    fn get_false_optional(&mut self) {
        /***
         * Probleme: 
         *      - Wie findet man den parent eines features?
         *      - Welches feature ist das root feature?
         * 
         * Beziehung eines features zu parent:
         * p => c (Damit das child ausgewählt werden kann muss der parent gewählt sein)
         * 
         * F_FM and p and not c is unsatisfiable
         * <=> #SAT(F_FM and p) == #SAT(F_FM and p and c) und f ist nicht mandatory
         * Wie ermittelt man ob ein feature mandatory ist?
         *      -> SAT(F_FM and p and other child von p)
         * 
        */
    }
}